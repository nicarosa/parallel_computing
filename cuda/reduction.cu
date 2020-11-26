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
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_width_step = width_input * channels_input;
    const int output_width_step = width_output * channels_output;

    if ((xIndex < width_output) && (yIndex < height_output))
    {
        int py = (int)(yIndex * y_ratio);
        int px = (int)(xIndex * x_ratio);
    
        float x_diff = (x_ratio * xIndex) - px;
        float y_diff = (y_ratio * yIndex) - py;
    
        uchar *ptr_img = input_image + (py * input_width_step);
        uchar *ptr_img_2 = input_image + ((py + 1) * input_width_step);

        for (int channel = 0; channel < channels_input; channel++)
	{
            int column = channels_input * px + channel;

            int pixel_value = *(ptr_img + column) * (1 - x_diff) * (1 - y_diff) +
                    *(ptr_img + column + channels_input) * x_diff * (1 - y_diff) +
                    *(ptr_img_2 + column) * (1 - x_diff) * y_diff + 
                    *(ptr_img_2 + column + channels_input) * x_diff * y_diff;
            *(output_image + (yIndex * output_width_step + xIndex * channels_output + channel)) = pixel_value;
        }
    }
}

int main(int argc, char* argv[]) 
{
    //parameters 1- source, 2- Destination , 3-threads, 4- algorithm-kind
	
    if (argc != 5) 
    {
        printf("Argumentos Incompletos.\n");
        exit(EXIT_FAILURE);
    }
	
    const string source_image_path = argv[1];
    const string result_image_path = argv[2];
    const int threads = atoi(argv[3]);
    const string algorithm = argv[4];
	
    //Cuda event para llevar los tiempos
    cudaEvent_t start, end;

    // Crear la imagen de 720x480 pixeles con 3 canales
    Mat output_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
    // Leer la imagen tomada del parametro source
    Mat input_image = imread(source_image_path);
    if(input_image.empty()) 
    {
        printf("Error con imagen.");
        exit(EXIT_FAILURE);
    }
    
    // Tamaño de matrices width/ancho * height/alto * 3
    const int input_bytes = input_image.cols * input_image.rows * input_image.channels() * sizeof(unsigned char);
    const int output_bytes = output_image.cols * output_image.rows * output_image.channels() * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    // Alloc la imagen de input
    cudaMalloc<unsigned char>(&d_input, input_bytes);
    // Alloc la imagen de output
    cudaMalloc<unsigned char>(&d_output, output_bytes);	
    //Copia la imagen de input del host localizada en la memoria del host a la imagen del input del dispositivo en su memoria 
    cudaMemcpy(d_input, input_image.ptr(), input_bytes, cudaMemcpyHostToDevice);
	
    // Time Management Start
    cudaEventCreate(&start);
    // Time Management End
    cudaEventCreate(&end);
    // Time Management Record
    cudaEventRecord(start, NULL);
    
    int channels_input = input_image.channels();
    int channels_output = output_image.channels();

    //Define los hilos por bloque
    const dim3 threadsPerBlock(threads, threads);
	
    // Calcula el tamaño de numBlocks para cubrir la imagen entera
    const dim3 numBlocks(width_output / threadsPerBlock.x, height_output / threadsPerBlock.y);
	
    //Corre el Kernel varias veces para medir un tiempo promedio.
    for(int i = 0; i < ITERATIONS; i++)
    {
            bilinear_scaling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_image.cols, input_image.rows, channels_input, output_image.cols, output_image.rows, channels_output);
    }

    // Time management Record Stop
    cudaEventRecord(end, NULL);

    // Time management Synchronize
    cudaEventSynchronize(end);

    //Calculo de tiempo entre eventos
    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, end);

    // Calcula e imprime tiempo real
    float time = elapsedTime / (ITERATIONS * 10.0f);
    printf(
        "Tiempo= %.8f s, Hilos/Block= %u\n",
        time,
        threadsPerBlock.x * threadsPerBlock.y
    );

    // Copia la imagen del dispositivo al host
    cudaMemcpy(output_image.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Copia la imagen a un file
    imwrite(result_image_path, output_image);

    // Libera la memoria global
    cudaFree(d_input);
    cudaFree(d_output);
    printf("Terminado\n");
	
    return 0;
}
