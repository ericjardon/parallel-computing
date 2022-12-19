%%writefile rectify.cu
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__
#endif

__global__ void rectifyPixel(unsigned char *image, unsigned char *new_image, int rows, int width, int height, int start)
{

    int offset = threadIdx.x * rows + start; // first row index in image

    for (int i = offset; i < offset + rows; i++) // rows
    {
        // Check height overflow
        // if (i == height) {
        //     break;
        // }
        for (int j = 0; j < width; j++) // pixels
        {
            int R = 4 * width * i + 4 * j + 0;
            int G = 4 * width * i + 4 * j + 1;
            int B = 4 * width * i + 4 * j + 2;
            int A = 4 * width * i + 4 * j + 3;

            if (image[R] < 127) {
                image[R] = 127;
            }
            if (image[G] < 127) {
                image[G] = 127;
            }
            if (image[B] < 127) {
                image[B] = 127;
            }
            if (image[A] < 127) {
                image[A] = 127;
            }

        }
    }
}

/*
Produces a rectified version of the input image, assigning the work
among N specified threads.

Assignment is done in the following way:
1. Divide the height of the image by the number of threads required to obtain K=the work per thread.
2. We launch the number of threads, each will work on K pixel rows of the image.
3. The remainder of this division are unassigned rows of pixels. This number is always less than N.
Hence, we assign a thread to each of these remainder rows. For this, we must launch twice.
*/
void process(char *input_filename, char *output_filename, int N)
{
    printf("Input file: %s\n", input_filename);
    printf("Output file: %s\n", output_filename);
    printf("Desired threads: %d\n", N);
    // 1. Allocate Host memory
    unsigned error;
    unsigned char *image, *new_image; // pointers to our PNGs
    unsigned width, height;

    // Load the image
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error)
        printf("error %u: %s\n", error, lodepng_error_text(error));

    printf("Image size: %d x %d ", width, height);
    size_t IMAGESIZE = width * height * 4 * sizeof(unsigned char);
    new_image = (unsigned char *)malloc(IMAGESIZE);

    // 2. Allocate Device Memory
    unsigned char *d_image, *d_new_image; // device copies of image and new image
    cudaMalloc((void **)&d_image, IMAGESIZE);
    cudaMalloc((void **)&d_new_image, IMAGESIZE);

    // 3. Memory transfer Host to Device
    cudaMemcpy(d_image, image, IMAGESIZE, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_image, new_image, IMAGESIZE, cudaMemcpyHostToDevice); unnecessary

    // Compute num of rows for each thread
    int rowsPerThread = height / N; // integer division floors to 0
    int remainder = height % N;  // number of remaining rows
    int lastOffset = height - remainder;
    printf("Rows per thread %d\n", rowsPerThread);
    printf("remainder %d\n", remainder);
    printf("lastOffset i=%d\n", lastOffset);

    // START timer
    float memsettime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    rectifyPixel<<<1, N>>>(d_image, d_new_image, rowsPerThread, width, height, 0);
    rectifyPixel<<<1, remainder>>>(d_image, d_new_image, 1, width, height, lastOffset); // Guaranteed remainder < N, 1 row per thread
    cudaDeviceSynchronize();

    // STOP cuda timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop);

    printf("*** CUDA execution time: %f ***\n", memsettime);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 4. Memory transfer back to Host
    cudaMemcpy(new_image, d_image, IMAGESIZE, cudaMemcpyDeviceToHost);

    lodepng_encode32_file(output_filename, new_image, width, height);

    cudaFree(d_image);
    cudaFree(d_new_image);

    free(image);
    free(new_image);
}

int main(int argc, char *argv[])
{
    // ./rectify <name of input png> <name of output png> < # threads>
    // process arguments
    int counter;

    if (argc != 4)
        printf("\nIncorrect number of arguments");
    if (argc >= 2)
    {
        printf("\nNumber Of Arguments Passed: %d", argc);
        printf("\n----Following Are The Command Line Arguments Passed----");
        for (counter = 0; counter < argc; counter++)
            printf("\nargv[%d]: %s", counter, argv[counter]);
    }

    int threads;
    sscanf(argv[3], "%d", &threads);

    process(argv[1], argv[2], threads);

    return 0;
}