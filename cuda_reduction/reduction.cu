#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <sys/time.h>

using namespace cv;
const int height = 480;
const int width = 854;

/*****************************************************************************
/*kernel
*****************************************************************************/

__global__ void DownSampleRGBAImageKernel(uint8_t *src_m, uint8_t *dst_m,
                                          int src_row, int src_col,
                                          int dst_row, int dst_col)
{

  int div = (width * height + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
  for (int l = 0; l < div; l++)
  {
    int ind = blockDim.x * blockIdx.x + threadIdx.x + l * (blockDim.x * gridDim.x);

    if (ind >= dst_row * dst_col)

      return;
    int cn = 3;

    int image_row = ind / dst_col;
    int image_col = ind % dst_col;

    float x_ratio = (src_col - 1) / (dst_col - 1);
    float y_ratio = (src_row - 1) / (dst_row - 1);

    uint8_t a, b, c, d, pixel;

    int x_l = floor(x_ratio * image_col), y_l = floor(y_ratio * image_row);
    int x_h = ceil(x_ratio * image_col), y_h = ceil(y_ratio * image_row);

    float x_weight = (x_ratio * image_col) - x_l;
    float y_weight = (y_ratio * image_row) - y_l;

    for (int k = 0; k < cn; k++)
    {
      a = src_m[y_l * src_col * cn + x_l * cn + k];
      b = src_m[y_l * src_col * cn + x_h * cn + k];
      c = src_m[y_h * src_col * cn + x_l * cn + k];
      d = src_m[y_h * src_col * cn + x_h * cn + k];

      pixel = (a & 0xff) * (1 - x_weight) * (1 - y_weight) + (b & 0xff) * x_weight * (1 - y_weight) + (c & 0xff) * y_weight * (1 - x_weight) + (d & 0xff) * x_weight * y_weight;

      dst_m[(image_row * dst_col + image_col) * cn + k] = pixel;
    }
  }
  //printf("Blue value: %d", pixelPtr[i*img.cols*cn +  j*cn + 0] );
}

/*****************************************************************************
/*Main
*****************************************************************************/
int main(int argc, char **argv)
{
  // *******************************************************  Vars initialization
  int blocksPerGrid, threadsPerBlock, totalThreads;

  std::string image_path = argv[1];
  std::string image_out_path = argv[2];

  int n_threads = atoi(argv[3]);
  int n_blocks = atoi(argv[4]);
  Mat img;
  uint8_t *resized, *d_resized, *d_img;

  cudaError_t err = cudaSuccess;

  struct timeval tval_before, tval_after, tval_result;

  // ******************************************************* get device info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int cores_mp = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

  /*printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
         deviceProp.multiProcessorCount,
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount); */


  if (n_threads != 0)
    blocksPerGrid = n_blocks;
  else
    blocksPerGrid = deviceProp.multiProcessorCount;

  if (n_threads != 0)
    threadsPerBlock = n_threads/n_blocks;
  else
    threadsPerBlock = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

  // 2;//(width * height + threadsPerBlock - 1) / threadsPerBlock;


  // *******************************************************  Read Matrix and declare resized matrix
  //printf("block,threads,time\n");

  img = imread(image_path, IMREAD_COLOR);
  if (img.empty())
  {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }

  resized = (uint8_t *)malloc(img.channels() * height * width * sizeof(uint8_t));

  //*******************************************************  device matrix declaration
  err = cudaMalloc((void **)&d_img, img.rows * img.step);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMalloc((void **)&d_resized, img.channels() * height * width * sizeof(uint8_t));

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // *******************************************************  copy from img to d_img
  uint8_t *pixelPtr = (uint8_t *)img.data;

  err = cudaMemcpy(d_img, pixelPtr, img.channels() * img.cols * img.rows * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // *******************************************************  Execute kernel
  gettimeofday(&tval_before, NULL); // get time

  dim3 dim_grid(blocksPerGrid);
  dim3 dim_block(threadsPerBlock);

  DownSampleRGBAImageKernel<<<dim_grid, dim_block>>>(
      d_img, d_resized, img.rows, img.cols, height, width);

  err = cudaGetLastError();

  // if (err == cudaSuccess)
  // {
  //   printf("All ok!");
  // }
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Time calculation
  gettimeofday(&tval_after, NULL);
  timersub(&tval_after, &tval_before, &tval_result);
  printf("%d,%d,%ld.%06ld\n", blocksPerGrid, threadsPerBlock * blocksPerGrid, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

  // *******************************************************  Copy d_resized to resized

  err = cudaMemcpy(resized, d_resized, img.channels() * height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // *******************************************************  Matrix convertion to Mat
  /*
  Mat resized_img(height, width, CV_8UC(3), resized);
  imshow("Display window", resized_img);
  int k = waitKey(0); // Wait for a keystroke in the window

  if (k == 's')
  {
    imwrite(image_out_path, resized_img);
  }
*/
  return 0;
}