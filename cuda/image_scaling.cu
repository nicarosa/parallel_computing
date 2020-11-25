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
#define ITERATIONS 20

// Function taken from https://github.com/sshniro/opencv-samples/blob/master/cuda-bgr-grey.cpp
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

/**
 * CUDA Kernel Device code
 *
 * Computes the new scaled output_image with NNS algorithm.
 */
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

    //2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    int px = 0, py = 0; 
    const int input_width_step = width_input * channels_input;
    const int output_width_step = width_output * channels_output;

    if ((xIndex < width_output) && (yIndex < height_output)){
        py = ceil(yIndex * y_ratio);
        px = ceil(xIndex * x_ratio);
        for (int channel = 0; channel < channels_output; channel++){
            *(output_image + (yIndex * output_width_step + xIndex * channels_output + channel)) =  *(input_image + (py * input_width_step + px * channels_input + channel));
        }
    }
}

/**
* CUDA Kernel Device code
*
* Implementation of Bilinear interpolation algorithm to down sample the source image.
*/
__global__ void bilinear_scaling(
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

    //2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_width_step = width_input * channels_input;
    const int output_width_step = width_output * channels_output;

    if ((xIndex < width_output) && (yIndex < height_output)){
        int py = (int)(yIndex * y_ratio);
        int px = (int)(xIndex * x_ratio);
    
        float x_diff = (x_ratio * xIndex) - px;
        float y_diff = (y_ratio * yIndex) - py;
    
        uchar *ptr_img = input_image + (py * input_width_step);
        uchar *ptr_img_2 = input_image + ((py + 1) * input_width_step);

        for (int channel = 0; channel < channels_input; channel++){
            int column = channels_input * px + channel;

            int pixel_value = *(ptr_img + column) * (1 - x_diff) * (1 - y_diff) +
                    *(ptr_img + column + channels_input) * x_diff * (1 - y_diff) +
                    *(ptr_img_2 + column) * (1 - x_diff) * y_diff + 
                    *(ptr_img_2 + column + channels_input) * x_diff * y_diff;
            *(output_image + (yIndex * output_width_step + xIndex * channels_output + channel)) = pixel_value;
        }
    }
}

/**
 * Host main routine
 */
int main(int argc, char* argv[]) {
    // Read parameters 1- source path, 2- Destination path, 3-threads, 4- algorithm
    if (argc != 5) {
        printf("Arguments are not complete. Usage: image_path image_result_path n_threads algorithm.\n");
        exit(EXIT_FAILURE);
    }
    const string source_image_path = argv[1];
    const string result_image_path = argv[2];
    const int threads = atoi(argv[3]);
    const string algorithm = argv[4];

    // time measurement variables
    cudaEvent_t start, end;

    // Create result image of 720x480 pixels with 3 channels
    Mat output_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
    // Read the image from the given source path
    Mat input_image = imread(source_image_path);
    if(input_image.empty()) {
        printf("Error reading image.");
        exit(EXIT_FAILURE);
    }
    
    // Matrices sizes width * height * 3
    const int input_bytes = input_image.cols * input_image.rows * input_image.channels() * sizeof(unsigned char);
    const int output_bytes = output_image.cols * output_image.rows * output_image.channels() * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    // Allocate the device input image
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input, input_bytes), "Failed to allocate device input image.");
    // Allocate the device output image
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output, output_bytes), "Failed to allocate device output image.");

    // Copy the host input image in host memory to the device input image in device memory
    SAFE_CALL(cudaMemcpy(d_input, input_image.ptr(), input_bytes, cudaMemcpyHostToDevice), "Failed to copy input image from host to device");

    // Create event to measure start time
    SAFE_CALL(cudaEventCreate(&start), "Failed to create start event.");

    // Create event to measure end time
    SAFE_CALL(cudaEventCreate(&end), "Failed to create end event");

    // Record the start event
    SAFE_CALL(cudaEventRecord(start, NULL), "Failed to start rescor of start event");
    
    int width_input = input_image.cols;
    int height_input = input_image.rows;
    int channels_input = input_image.channels();
    int width_output = output_image.cols;
    int height_output = output_image.rows;
    int channels_output = output_image.channels();

    const dim3 threadsPerBlock(threads, threads);
    //Calculate numBlocks size to cover the whole image        
    const dim3 numBlocks(width_output / threadsPerBlock.x, height_output / threadsPerBlock.y);

    // Run kernel several times to measure an average time.
    for(int i = 0; i < ITERATIONS; i++){
        if(algorithm == "Nearest") {
            nearest_neighbour_scaling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width_input, height_input, channels_input, width_output, height_output, channels_output);
        } else if(algorithm == "Bilinear") {
            bilinear_scaling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width_input, height_input, channels_input, width_output, height_output, channels_output);
        }
        SAFE_CALL(cudaGetLastError(), "Failed to launch kernel");
    }

    // Record the stop event
    SAFE_CALL(cudaEventRecord(end, NULL), "Failed to record end event.");

    // Wait for the stop event to complete
    SAFE_CALL(cudaEventSynchronize(end), "Failed to synchronize on the end event");

    float msecTotal = 0.0f;
    SAFE_CALL(cudaEventElapsedTime(&msecTotal, start, end), "Failed to get time elapsed between events");

    // Compute and print the performance
    float secPerMatrixMul = msecTotal / (ITERATIONS * 1000.0f);
    double flopsPerMatrixMul = 2.0 * (double)width_output * (double)height_output * channels_output;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (secPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.8f s, Size= %.0f Ops, WorkgroupSize= %u threads/block, Blocks= %u\n",
        gigaFlops,
        secPerMatrixMul,
        flopsPerMatrixMul,
        threadsPerBlock.x * threadsPerBlock.y,
        numBlocks.x * numBlocks.y
    );

    // Copy the device output image in device memory to the host output image in host memory.
    SAFE_CALL(cudaMemcpy(output_image.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost), "Failed to copy output image from device to host");

    // Write the image to a file
    imwrite(result_image_path, output_image);

    // Free device global memory
    SAFE_CALL(cudaFree(d_input), "Failed to free device input image");
    SAFE_CALL(cudaFree(d_output), "Failed to free device output image");

    printf("Done\n");
    return 0;
}

