#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define N 512     // square matrix size
#define N2 262144   // N*N number of elements

#define n 0.0002
#define p 0.5
#define G 0.75

// Assignment combinations
// A) 1024 threads/block and 16 blocks can allow each thread to handle 16 finite elements

#define N_BLOCKS  32
#define N_THREADS_BLOCK 64

/* Print the grid for debugging purposes.*/
void printGrid(float* m) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%e\t", m[i * N + j]);
        }
        printf("\n");
    }
}

/**
 * @brief
 * Wrapper to catch any CUDA API errors and exit.
 */
#define gpuErrorCheck(ans)                    \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
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
 * Return index in a 2d flattened array given row and col indexes.
 */
__device__ int lin(int x, int y) {
    return x * N + y % N;
}

void updateDeviceRefs(float *&d_u, float *&d_u1, float *&d_u2) {
    cudaFree(d_u2);
	d_u2 = d_u1;    // d_u2 now points to d_u1
	d_u1 = d_u;    // d_u1 now points to d_u
}

__global__ void updateCorners(float* u) {
    // No partition applicable
    if (threadIdx.x==0) {
        u[lin(0, 0)] = G * u[lin(1, 0)];
    }
    else if (threadIdx.x==1) {
        u[lin(N - 1, 0)] = G * u[lin(N - 2, 0)];
    }
    else if (threadIdx.x==2) {
        u[lin(0, N - 1)] = G * u[lin(0, N - 2)];
    }
    else if (threadIdx.x==3) {
        u[lin(N - 1, N - 1)] = G * u[lin(N - 1, N - 2)];
    }
}

__global__ void updateSides(float* u, float* u1, float* u2, int partitionSize) {
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * partitionSize;
    int index, row, col;
    bool edgeX, edgeY;
    for (int i=0; i<partitionSize; i++) {
        index = i + offset;
        if (!(index < N2)) break;

        row = (index) / N;
        col = (index) % N;
        edgeX = (row == 0 || row == N - 1);
        edgeY = (col == 0 || col == N - 1);

        if (!(edgeX && edgeY)) {
            if (edgeX) {
                    if (row == 0) {
                        u[lin(0, col)] = G * u[lin(1, col)];
                    }
                    else {
                        u[lin(N - 1, col)] = G * u[lin(N - 2, col)];
                    }
                }
            else if (edgeY) {
                if (col == 0) {
                    u[lin(row, 0)] = G * u[lin(row, 1)];
                }
                else {
                    u[lin(row, N - 1)] = G * u[lin(row, N - 2)];
                }
            }
        }
    }

}

__global__ void fillMatrix(float* matrix, float midpoint) {
    int x = blockIdx.x;
    int y = threadIdx.x;

    int index = lin(x,y);

    if (x == N/2 && y == N/2) {
        // printf("Adding 1 to %d\n", index);
        matrix[index] = midpoint;
    } else {
        matrix[index] = 0.0;
    }
}

__global__ void updateInteriors(float* u, float* u1, float* u2, int partitionSize) {
    // We use two indexes: block and thread, analogous to row and column, 
    // compute starting index: (row_i*width + col_i) + offset
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * partitionSize;
    int index, row, col;

    // Every thread works on partitionSize elements of the grid
    for (int i = 0; i < partitionSize; i++) {
        index = i + offset;
        if (!(index < N2)) break;

        row = (index) / N;
        col = (index) % N;
        // Skip sides and corners
        if (row == 0 || row == N-1 || col == 0 || col == N-1) {
            continue;
        }
        // Is interior element
        u[lin(row, col)] = ((p * (u1[lin(row - 1, col)] + u1[lin(row + 1, col)] + u1[lin(row, col - 1)] + u1[lin(row, col + 1)] - 4 * u1[lin(row, col)])) +
            (2 * u1[lin(row, col)]) - ((1 - n) * u2[lin(row, col)])) / (1 + n);
    }
} // note: boundary conditions, are they partition-wise????

/**
 * @brief Implements a 512x512 finite element grid
 *
 * @param N the number of rows and columns of grid
 * @param n_iterations the number of iterations to run simulation
 */
void synthesizeGPU(int n_iterations) {
    
    float *u;
    size_t SIZE = N2 * sizeof(float);
    u = (float*) malloc(SIZE);
    cudaError_t cudaStatus;

    // GPU Memory Allocation
    float *d_u , *d_u1, *d_u2;
    gpuErrorCheck(cudaMalloc((void**)&d_u, SIZE));
    gpuErrorCheck(cudaMalloc((void**)&d_u1, SIZE));
    gpuErrorCheck(cudaMalloc((void**)&d_u2, SIZE));

    // Initialize u to 0.0
    fillMatrix<<<N, N>>>(d_u, 0);
    // Initial non-zero value at u1(N/2, N/2)
    fillMatrix<<<N, N>>>(d_u1, 1.0);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Matrix init failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

    // DECOMPOSITION
    int capacity = N_BLOCKS * N_THREADS_BLOCK;
    int elemsPerThread = N2 / capacity + (int)((N2 % capacity) > 0); // each thread will do +1 if there is a remainder
    
    printf("Blocks: %d\n", N_BLOCKS);
    printf("Threads per block: %d\n", N_THREADS_BLOCK);
    printf("Elements per thread: %d\n", elemsPerThread);
    
    // Start timer
    float memsettime;
    cudaEvent_t start, stop;
    gpuErrorCheck(cudaEventCreate(&start));
    gpuErrorCheck(cudaEventCreate(&stop));
    gpuErrorCheck(cudaEventRecord(start, 0));
    
    for (int iter = 0; iter < n_iterations; iter++) {
        // 1. Interior Elements
        updateInteriors << <N_BLOCKS, N_THREADS_BLOCK>> > (d_u, d_u1, d_u2, elemsPerThread);
        // 2. Sides
        updateSides << <N_BLOCKS, N_THREADS_BLOCK>> > (d_u, d_u1, d_u2, elemsPerThread);
        // 3. Corners
        updateCorners<<<1, 4>>>(d_u);

        cudaDeviceSynchronize();

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Matrix update failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }

        // Shift matrices
        updateDeviceRefs(d_u, d_u1, d_u2);

        // Allocate new d_u
        gpuErrorCheck(cudaMalloc((void**)&d_u, SIZE));

        // Copy iteration results to CPU
        gpuErrorCheck(cudaMemcpy(u, d_u1, SIZE, cudaMemcpyDeviceToHost));
        
        // Record result
        printf("%d:\t u(256, 256) = %.6f\n", iter, u[256 * N + 256 % N]);
    }

    // Stop timer
    gpuErrorCheck(cudaEventRecord(stop, 0));
    gpuErrorCheck(cudaEventSynchronize(stop));
    gpuErrorCheck(cudaEventElapsedTime(&memsettime, start, stop));
    printf("*** CUDA execution time: %f ***\n", memsettime);

    gpuErrorCheck(cudaEventDestroy(start));
    gpuErrorCheck(cudaEventDestroy(stop));

    cudaFree(d_u);
    cudaFree(d_u1);
    cudaFree(d_u2);
    free(u);
    printf("\n End of Simulation");
}


int main(int argc, char* argv[])
{
    // ============
    // RUN COMMAND:
    // ./grid_4_4 <number_of_iterations> 
    // ============

    if (argc != 2) {
        printf("\nIncorrect arguments. Run the program as follows:\n");
        printf("./grid_4_4 <number_of_iterations>\n");
    }

    int T;
    sscanf(argv[1], "%d", &T);

    // Simulate Music Synthesis by finite element on 4x4 grid
    synthesizeGPU(T);

    return 0;
}