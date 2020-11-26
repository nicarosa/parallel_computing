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

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

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

int main(int argc, char* argv[]) {
    //parameters 1- source, 2- Destination , 3-threads, 4- algorithm-kind
    if (argc != 5) {
        printf("Arguments are not complete. Usage: image_path image_result_path n_threads algorithm.\n");
        exit(EXIT_FAILURE);
    }
    const string source_image_path = argv[1];
    const string result_image_path = argv[2];
    const int threads = atoi(argv[3]);
    const string algorithm = argv[4];

    cudaEvent_t start, end;

    // Crear la imagen de 720x480 pixeles con 3 canales
    Mat output_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
    // Leer la imagen tomada del parametro source
    Mat input_image = imread(source_image_path);
    if(input_image.empty()) {
        printf("Error reading image.");
        exit(EXIT_FAILURE);
    }
    
    // Tamaño de matrices width/ancho * height/alto * 3
    const int input_bytes = input_image.cols * input_image.rows * input_image.channels() * sizeof(unsigned char);
    const int output_bytes = output_image.cols * output_image.rows * output_image.channels() * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    // Alloc la imagen de input
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input, input_bytes), "Failed to allocate device input image.");
    // Alloc la imagen de output
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output, output_bytes), "Failed to allocate device output image.");
	
    //Copia la imagen de input del host localizada en la memoria del host a la imagen del input del dispositivo en su memoria 
    SAFE_CALL(cudaMemcpy(d_input, input_image.ptr(), input_bytes, cudaMemcpyHostToDevice), "Failed to copy input image from host to device");

    // Time Management Start
    SAFE_CALL(cudaEventCreate(&start), "Failed to create start event.");

    // Time Management End
    SAFE_CALL(cudaEventCreate(&end), "Failed to create end event");

    // Time Management Record
    SAFE_CALL(cudaEventRecord(start, NULL), "Failed to start rescor of start event");
    
    int width_input = input_image.cols;
    int height_input = input_image.rows;
    int channels_input = input_image.channels();
    int width_output = output_image.cols;
    int height_output = output_image.rows;
    int channels_output = output_image.channels();

    const dim3 threadsPerBlock(threads, threads);
	
    // Calcula el tamaño de numBlocks para cubrir la imagen entera
    const dim3 numBlocks(width_output / threadsPerBlock.x, height_output / threadsPerBlock.y);
	
    //Corre el Kernel varias veces para medir un tiempo promedio.
    for(int i = 0; i < ITERATIONS; i++){
        if(algorithm == "Nearest") {
            nearest_neighbour_scaling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width_input, height_input, channels_input, width_output, height_output, channels_output);
        } else if(algorithm == "Bilinear") {
            bilinear_scaling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width_input, height_input, channels_input, width_output, height_output, channels_output);
        }
        SAFE_CALL(cudaGetLastError(), "Failed to launch kernel");
    }

    // Time management Record Stop
    SAFE_CALL(cudaEventRecord(end, NULL), "Failed to record end event.");

    // Time management Synchronize
    SAFE_CALL(cudaEventSynchronize(end), "Failed to synchronize on the end event");

    float msecTotal = 0.0f;
    SAFE_CALL(cudaEventElapsedTime(&msecTotal, start, end), "Failed to get time elapsed between events");

    // Calcula e imprime tiempo
    float secPerMatrixMul = msecTotal / (ITERATIONS * 1000.0f);
    printf(
        "Time= %.8f s, WorkgroupSize= %u threads/block, Blocks= %u\n",
        secPerMatrixMul,
        threadsPerBlock.x * threadsPerBlock.y,
        numBlocks.x * numBlocks.y
    );

    // Copia la imagen del dispositivo al host
    SAFE_CALL(cudaMemcpy(output_image.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost), "Failed to copy output image from device to host");

    // Copia la imagen a un archivo
    imwrite(result_image_path, output_image);

    // Libera la memoria global
    SAFE_CALL(cudaFree(d_input), "Failed to free device input image");
    SAFE_CALL(cudaFree(d_output), "Failed to free device output image");

    printf("Terminado\n");
    return 0;
}

