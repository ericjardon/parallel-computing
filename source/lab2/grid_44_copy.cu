%%writefile grid_4_4.cu
#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants
#define N 4
#define N2 16

#define n 0.0002
#define p 0.5
#define G 0.75

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

__global__ void updateSides(float* u, float* u1, float* u2) {
    int x = threadIdx.x;
    int y = threadIdx.y;

    bool edgeX = (x == 0 || x == N - 1);
    bool edgeY = (y == 0 || y == N - 1);

    if (edgeX && edgeY) {
        return;
    }

    if (edgeX) {
        if (x == 0) {
            u[lin(0, y)] = G * u[lin(1, y)];
        }
        else {
            u[lin(N - 1, y)] = G * u[lin(N - 2, y)];
        }
    }
    else if (edgeY) {
        if (y == 0) {
            u[lin(x, 0)] = G * u[lin(x, 1)];
        }
        else {
            u[lin(x, N - 1)] = G * u[lin(x, N - 2)];
        }
    }
}

__global__ void updateInteriors(float* u, float* u1, float* u2) {
    int x = threadIdx.x;
    int y = threadIdx.y;

    bool edgeX = (x == 0 || x == N - 1);
    bool edgeY = (y == 0 || y == N - 1);

    if (!edgeX && !edgeY) {
        // Interior Elements
        u[lin(x, y)] = ((p * (u1[lin(x - 1, y)] + u1[lin(x + 1, y)] + u1[lin(x, y - 1)] + u1[lin(x, y + 1)] - 4 * u1[lin(x, y)])) +
            (2 * u1[lin(x, y)]) - ((1 - n) * u2[lin(x, y)])) / (1 + n);
    }
}

/**
 * @brief Implements a 4 by 4 finite element grid
 *
 * @param N the number of rows and columns of grid
 * @param n_iterations the number of iterations to run simulation
 */
void synthesizeGPU(int n_iterations) {

    // Initialize three 4x4 grids to 0
    float u[N2] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };
    float u1[N2] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };
    float u2[N2] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    // Initial non-zero value at u1(2,2)
    u1[2 * N + 2 % N] = 1.0;

    // GPU Memory Allocation
    size_t SIZE = N2 * sizeof(float);
    float *d_u , *d_u1, *d_u2;
    gpuErrorCheck(cudaMalloc((void**)&d_u, SIZE));
    gpuErrorCheck(cudaMalloc((void**)&d_u1, SIZE));
    gpuErrorCheck(cudaMalloc((void**)&d_u2, SIZE));

    // Copy initialized matrices CPU to Device
    gpuErrorCheck(cudaMemcpy(d_u, u, SIZE, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_u1, u1, SIZE, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_u2, u2, SIZE, cudaMemcpyHostToDevice));
    
    // Use NxN threads with 2d index. For every cell, one thread.
    dim3 dimBlock(N, N);

    // Simulate for iter iterations
    for (int iter = 0; iter < n_iterations; iter++) {
        // 1. Update Interior Elements
        updateInteriors << <1, dimBlock >> > (d_u, d_u1, d_u2);
        // 2. Update Sides
        updateSides << <1, dimBlock >> > (d_u, d_u1, d_u2);
        // 3. Update Corners
        updateCorners<<<1, 4>>>(d_u);

        cudaDeviceSynchronize();

        // Shift matrices
        updateDeviceRefs(d_u, d_u1, d_u2);

        // Next d_u matrix
        gpuErrorCheck(cudaMalloc((void**)&d_u, SIZE));

        // Copy iteration results to CPU
        gpuErrorCheck(cudaMemcpy(u, d_u1, SIZE, cudaMemcpyDeviceToHost));
        //printf("%d:\t u(2,2) = %e\n", iter, u[2 * N + 2 % N]);
        printGrid(u);
        printf("\n");
    }

    printf("\n End of Simulation");
    cudaFree(d_u);
    cudaFree(d_u1);
    cudaFree(d_u2);

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