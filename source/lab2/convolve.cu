#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include "wm.h"
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__
#endif

#define R 0
#define G 1
#define B 2
#define A 3

/**
 * @brief
 * Wrapper to catch any CUDA API errors
 * Will exit the program if an error is caught.
 */
#define gpuErrorCheck(ans)                    \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
    inline void
    gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/**
 * @brief
 * Clamps a weighted sum value between 255 and 0
 * @param wsum
 * @return float
 */
__device__ float clampValue(float wsum)
{
    if (wsum < 0)
    {
        return 0;
    }
    if (wsum > 255)
    {
        return 255;
    }
    return wsum;
}

/**
 * @brief
 * Single thread work for producing N/T weighted sums and writing to the output image pixels
 * @param input the input image array
 * @param output the output image array
 * @param width the width of the image in pixels
 * @param height the height of the image in pixels
 * @param T number of threads to launch
 * @param weights the pointer to a 3x3 matrix to use for convolution (a.k.a. kernel)
 * @return void
 */
__global__ void convolutionGPU(unsigned char *input, unsigned char *output, float *weights, int inputWidth, int inputHeight, int T, int pixelsPerThread, int totalPixels)
{
    // This thread should compute output for each pixel i,j
    int pixelOffset = threadIdx.x * pixelsPerThread; // assign work to each thread in contiguous chunks

    int outputWidth = inputWidth - 2;
    int counter = 0;
    for (int i = 0; i < pixelsPerThread; i++)
    {

        // compute pixel index in original image as starting pixel
        // int pixelIndex = ((pixelOffset + i) / outputWidth)*(inputWidth) + ((pixelOffset + i) % outputWidth);
        int pixelRow = (pixelOffset + i) / inputWidth;
        int pixelCol = (pixelOffset + i) % inputWidth;

        // Ghost cells
        if (pixelCol == 0 || pixelCol == inputWidth - 1 || pixelRow == 0 || pixelRow == inputHeight - 1)
        {
            counter += 1;
            continue;
        }

        // Compute weighted sum of 3 color channels
        float sums[3] = {0, 0, 0};
        // Sweep through every pixel in the 3x3 tile with pixelRow, pixelCol as center
        for (int ii = 0; ii < 3; ii++)
        {
            for (int jj = 0; jj < 3; jj++)
            {
                // Row and column in 2d image
                int r = pixelRow + ii - 1;
                int c = pixelCol + jj - 1;
                // Map to a 1d index
                int pi = 4 * r * inputWidth + 4 * c;
                float w = weights[3 * ii + jj];
                
                sums[R] += (w * input[pi + R]);
                sums[G] += (w * input[pi + G]);
                sums[B] += (w * input[pi + B]);
            }
        }

        // Map to output index; offset by -1 on both axes and use output width to place in row
        int outPixel = (4 * (pixelRow - 1)) * outputWidth + (4 * (pixelCol - 1));

        output[outPixel + R] = (unsigned char) round(clampValue(sums[R]));
        output[outPixel + G] = (unsigned char) round(clampValue(sums[G]));
        output[outPixel + B] = (unsigned char) round(clampValue(sums[B]));
        // Alpha should be the same as input pixel?
        output[outPixel + A] = input[4 * pixelRow*inputWidth + 4 * pixelCol  + A]; // alpha channel
    }
}

/**
 * @brief Produces a convolved version of the given PNG file using the weights defined in "wm.h",
 * writing the results to the output matrix.
 **/
int convolveImage(char *input_filename, char *output_filename, int N_threads)
{
    printf("Input PNG: %s\n", input_filename);
    printf("Output PNG: %s\n", output_filename);
    printf("Threads: %d\n", N_threads);

    unsigned error;
    unsigned char *image, *output_image; // pointers to our PNGs
    unsigned width, height;
    // Load image
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error)
    {
        printf("Decode error %u: %s\n", error, lodepng_error_text(error));
        return -1;
    }
    printf("Input size: %d x %d \n", width, height);
    printf("Output size: %d x %d \n", width - 2, height - 2);

    size_t INPUTSIZE = width * height * 4 * sizeof(unsigned char);
    size_t OUTPUTSIZE = (width - 2) * (height - 2) * 4 * sizeof(unsigned char);

    // Allocate cpu memory
    output_image = (unsigned char *)malloc(OUTPUTSIZE);

    // Allocate gpu Memory
    unsigned char *d_input;
    unsigned char *d_output;
    float *d_weights;

    gpuErrorCheck(cudaMalloc((void **)&d_input, INPUTSIZE));
    gpuErrorCheck(cudaMalloc((void **)&d_output, OUTPUTSIZE));
    gpuErrorCheck(cudaMalloc((void **)&d_weights, 9 * sizeof(float)));

    gpuErrorCheck(cudaMemcpy(d_input, image, INPUTSIZE, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_output, output_image, OUTPUTSIZE, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_weights, w, 9 * sizeof(float), cudaMemcpyHostToDevice)); // w[0], &w

    // Start timer
    float memsettime;
    cudaEvent_t start, stop;
    gpuErrorCheck(cudaEventCreate(&start));
    gpuErrorCheck(cudaEventCreate(&stop));
    gpuErrorCheck(cudaEventRecord(start, 0));

    // Split evenly the number of pixels in output image among N_threads
    int size = (width) * (height);
    int sums_per_thread = (size) / N_threads;
    // int N_remainder = (size) % N_threads;
    printf("# pixels per thread = %d\n", sums_per_thread);

    convolutionGPU<<<1, N_threads>>>(d_input, d_output, d_weights, width, height, N_threads, sums_per_thread, size);
    // convolutionGPU<<<1, N_remainder>>>(d_input, d_output, d_weights, width, height, N_threads, sums_per_thread, size);
    cudaDeviceSynchronize();

    // Stop timer
    gpuErrorCheck(cudaEventRecord(stop, 0));
    gpuErrorCheck(cudaEventSynchronize(stop));
    gpuErrorCheck(cudaEventElapsedTime(&memsettime, start, stop));
    printf("*** CUDA execution time: %f ***\n", memsettime);

    gpuErrorCheck(cudaEventDestroy(start));
    gpuErrorCheck(cudaEventDestroy(stop));

    // 4. Memory transfer back to Host
    gpuErrorCheck(cudaMemcpy(output_image, d_output, OUTPUTSIZE, cudaMemcpyDeviceToHost));

    error = lodepng_encode32_file(output_filename, output_image, width - 2, height - 2);
    if (error)
    {
        printf("Encode error: %u: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    free(image);
    free(output_image);
    gpuErrorCheck(cudaFree(d_input));
    gpuErrorCheck(cudaFree(d_output));
    gpuErrorCheck(cudaFree(d_weights));
    return 0;
}

int main(int argc, char *argv[])
{
    // Process arguments
    int counter;

    if (argc != 4)
    {
        printf("\nIncorrect number of arguments. Please run the program as follows:\n");
        printf(".\\convolve <input image.png> <output image.png> <num of threads> \n");
    }

    if (argc >= 2)
    {
        printf("\nNumber Of Arguments Passed: %d", argc);
        printf("\n----Following Are The Command Line Arguments Passed----");
        for (counter = 0; counter < argc; counter++)
            printf("\nargv[%d]: %s", counter, argv[counter]);
    }

    int threads;
    sscanf(argv[3], "%d", &threads);

    // Assign tasks and launch kernels accordingly
    convolveImage(argv[1], argv[2], threads);

    return 0;
}