// #include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define RESULT_WIDTH 720
#define RESULT_HEIGHT 480
#define CHANNELS 3
#define ITERATIONS 20

//function to get time
#ifndef TIMER_H
#define TIMER_H

typedef unsigned long long timestamp_t;

static timestamp_t

get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
#endif

Mat output_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3); 
Mat input_image;

__global__ void nearest_neighbour_scaling(
    unsigned char *input_image, 
    unsigned char *output_image,
    int width_input, 
    int height_input,
    int channels_input,
    int width_output, 
    int height_output,
    int channels_output) {
    const float x_ratio = (width_input + 0.0) / width_output;
    const float y_ratio = (height_input + 0.0) / height_output;

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    int px = 0, py = 0; 
    const int input_width_step = width_input * channels_input;
    const int output_width_step = width_output * channels_output;

    if ((xIndex < width_output) && (yIndex < height_output)){
        py = ceil(yIndex * y_ratio);
        px = ceil(xIndex * x_ratio);
        for (int channel = 0; channel < CHANNELS; channel++){
            *(output_image + (yIndex * output_width_step + xIndex * channels_output + channel)) =  *(input_image + (py * input_width_step + px * channels_input + channel));
        }
    }
}



int main(int argc, char* argv[]) {
    
    /*Parameter inputs: Source, Out and number of threads*/
    const string src = argv[1];
    const string dst = argv[2];
    const int threads = atoi(argv[3]);

    input_image = imread(src);

    /*Allocate Space*/
    const int size_input = sizeof(unsigned char) * input_image.cols * input_image.rows * CHANNELS; 
    const int size_output = sizeof(unsigned char) *output_image.cols * output_image.rows * CHANNELS;
    unsigned char *input_image_pointer, *output_image_pointer;

    cudaMalloc<unsigned char>(&input_image_pointer, size_input);
    cudaMalloc<unsigned char>(&output_image_pointer, size_output);
    cudaMemcpy(input_image_pointer, input_image.ptr(), size_input, cudaMemcpyHostToDevice);

    /*Time Recording*/
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, NULL);

   
    const dim3 threadsPerBlock(threads, threads);
    const dim3 numBlocks(output_image.cols / threadsPerBlock.x, output_image.rows / threadsPerBlock.y);
    for(int i = 0; i < ITERATIONS; i++){
        nearest_neighbour_scaling<<<numBlocks, threadsPerBlock>>>(input_image_pointer, output_image_pointer, input_image.cols, input_image.rows, CHANNELS, output_image.cols, output_image.rows, CHANNELS);
    }

    /*Stop Time recording*/
    cudaEventRecord(end, NULL);
    cudaEventSynchronize(end);
    
    /*Calculate Time*/
    float time = 0.0f;
    cudaEventElapsedTime(&time, start, end);
    float totalTime = time / (ITERATIONS * 1.0f);
    printf("%.8f",totalTime);
  
    cudaMemcpy(output_image.ptr(), output_image_pointer, size_output, cudaMemcpyDeviceToHost);

    imwrite(dst, output_image);

    cudaFree(input_image_pointer);
    cudaFree(output_image_pointer);

    return 0;
}
