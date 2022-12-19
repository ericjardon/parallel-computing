%%writefile pool.cu
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__
#endif

/*
TODO: review indexes in out image buffer.
- last square is not reached
- we may need to multiply by 2 or 4 somewhere
*/
__global__ void maxPool(unsigned char *image, unsigned char *compressed_image, int totalSquares, int squaresPerRow, int T, int width)
{
    // Thread is not needed
    if (threadIdx.x > totalSquares) {
        return;
    }

    // Work on a square every T squares, find max of every channel RGBA among 4 pixels in square
    int i = 0;
    while (1) {
        int S = i*T + threadIdx.x;
        if (S > totalSquares) {
            return;
        };
        // compute row and column of pixel
        int square_row = (S / squaresPerRow);
        int square_col = (S % squaresPerRow);
        // look at all four pixels in the square
        unsigned char max_R = 0;
        unsigned char max_G = 0;
        unsigned char max_B = 0;
        unsigned char max_A = 0;

        // Find max of every channel R G B A
        for (int delta_row = 0; delta_row < 2; delta_row++) { // 0, 1
            for (int delta_col = 0; delta_col < 2; delta_col++) { // 0, 1
                // Individual pixels' coordinates
                int i = square_row * 2 + delta_row;
                int j = square_col * 2 + delta_col;

                // Translate coords to entries in 1d array
                int R = 4 * width * i + 4 * j + 0;
                int G = 4 * width * i + 4 * j + 1;
                int B = 4 * width * i + 4 * j + 2;
                int A = 4 * width * i + 4 * j + 3;

                if (image[R] > max_R) {
                    max_R = image[R];
                }
                if (image[G] > max_G) {
                    max_G = image[G];
                }
                if (image[B] > max_B) {
                    max_B = image[B];
                }
                if (image[A] > max_A) {
                    max_A = image[A];
                }
            }
        }

        int R_ = 4 * (squaresPerRow) * square_row + 4 * square_col + 0;
        int G_ = 4 * (squaresPerRow) * square_row + 4 * square_col + 1;
        int B_ = 4 * (squaresPerRow) * square_row + 4 * square_col + 2;
        int A_ = 4 * (squaresPerRow) * square_row + 4 * square_col + 3;

        compressed_image[R_] = max_R;
        compressed_image[G_] = max_G;
        compressed_image[B_] = max_B;
        compressed_image[A_] = max_A;
        i++;
    }
}

/*
Produces a compressed version of the input image
by performing 2x2 max pooling with a desired number 
of threads (assumes equal width and height).

Max pooling outputs an N/2 x N/2 image,
keeps the max value from every channel in 
chunks of 2x2 pixels.

Thread assignment is done in the following way:
0. Split the image into 2x2 squares. Assume w=h, track number of total squares: S
1. Each of the T threads will work on a square every T squares.
    Mathematically, a thread will have have every T-th square. so squareId = i*T + threadId while  i < total squares
2. With odd image lengths, the last row or column of pixels is ommitted from the output.
*/
void process(char *input_filename, char *output_filename, int N)
{
    printf("Input file: %s\n", input_filename);
    printf("Output file: %s\n", output_filename);
    printf("Desired threads: %d\n", N);
    // 1. Allocate Host memory
    unsigned error;
    unsigned char *image, *compressed_image; // pointers to our PNGs
    unsigned width, height;

    // Load the image
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error)
        printf("error %u: %s\n", error, lodepng_error_text(error));

    printf("Image size: %d x %d \n", width, height);
    printf("Output size: %d x %d \n", width/2, height/2);

    size_t IMAGESIZE = width * height * 4 * sizeof(unsigned char);

    size_t COMPRESSEDSIZE = (width/2) * (height/2) * 4 * sizeof(unsigned char);
    compressed_image = (unsigned char *)malloc(COMPRESSEDSIZE);

    // 2. Allocate Device Memory
    unsigned char *d_image; // device copies of image and new image
    unsigned char *d_compressed_image; // device copies of image and new image
    cudaMalloc((void **)&d_image, IMAGESIZE);
    cudaMalloc((void **)&d_compressed_image, COMPRESSEDSIZE);

    // 3. Memory transfer Host to Device
    cudaMemcpy(d_image, image, IMAGESIZE, cudaMemcpyHostToDevice);

    // Compute num of squares for each thread
    int squaresPerRow = width / 2;
    int totalSquares = (squaresPerRow)*(height/2); // assume width == height
    printf("Total Squares %d", totalSquares);
    // START timer
    float memsettime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    maxPool<<<1, N>>>(d_image, d_compressed_image, totalSquares, squaresPerRow, N, width);
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
    cudaMemcpy(compressed_image, d_compressed_image, COMPRESSEDSIZE, cudaMemcpyDeviceToHost);

    lodepng_encode32_file(output_filename, compressed_image, width/2, height/2);

    cudaFree(d_image);
    cudaFree(d_compressed_image);

    free(image);
    free(compressed_image);
}

int main(int argc, char *argv[])
{
    // ./pool <name of input png> <name of output png> < # threads>
    // process arguments
    int counter;

    if (argc != 4) {
        printf("\nIncorrect number of arguments. Please provide arguments like so:\n");
        printf(".\\pool <Input image.png> <Output image.png> <Number of threads> \n");
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

    process(argv[1], argv[2], threads);

    return 0;
}