%%writefile grid_4_4.cu
#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 4
#define N2 16

void printGrid(double* m) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%e\t", m[i * N + j]);
        }
        printf("\n");
    }
}

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
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/*
Returns the linearized index for two coordinates: x - row, y - col
*/
__device__ int lin(int x, int y) {
    return x * N + y % N;
}

int linCpu(int x, int y) {
    return x * N + y % N;
}

void updateCorners(double* u, double G) {
    // Update corners serially; as we get no gain from parallelizing 4 elements
    u[linCpu(0, 0)] = G * u[linCpu(1, 0)];
    u[linCpu(N - 1, 0)] = G * u[linCpu(N - 2, 0)];
    u[linCpu(0, N - 1)] = G * u[linCpu(0, N - 2)];
    u[linCpu(N - 1, N - 1)] = G * u[linCpu(N - 1, N - 2)];
}

__global__ void updateSides(double* u, double* u1, double* u2, double n, double p, double G) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    // printf("tid (%d, %d) idx=%d \n",x,y, lin(x,y));
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

__global__ void updateInterior(double* u, double* u1, double* u2, double n, double p, double G) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    // printf("tid (%d, %d) idx=%d \n",x,y, lin(x,y));
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
    double n = 0.0002;
    double p = 0.5;
    double G = 0.75;

    double u[N2] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };
    double u1[N2] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };
    double u2[N2] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    // Initial value
    u1[2 * N + 2 % N] = 1.0;

    // GPU Memory
    size_t SIZE = N2 * sizeof(double);
    double* d_u;
    double* d_u1;
    double* d_u2;

    gpuErrorCheck(cudaMalloc((void**)&d_u, SIZE));
    gpuErrorCheck(cudaMalloc((void**)&d_u1, SIZE));
    gpuErrorCheck(cudaMalloc((void**)&d_u2, SIZE));
    // CUDA uses 1d linearized arrays
    dim3 dimBlock(N, N);

    for (int iter = 0; iter < n_iterations; iter++) {
        // Copy CPU to Device
        gpuErrorCheck(cudaMemcpy(d_u, u, SIZE, cudaMemcpyHostToDevice));
        gpuErrorCheck(cudaMemcpy(d_u1, u1, SIZE, cudaMemcpyHostToDevice));
        gpuErrorCheck(cudaMemcpy(d_u2, u2, SIZE, cudaMemcpyHostToDevice));

        // 1. Update Interior Elements
        updateInterior << <1, dimBlock >> > (d_u, d_u1, d_u2, n, p, G);
        cudaDeviceSynchronize();
        // 2. Update Sides
        updateSides << <1, dimBlock >> > (d_u, d_u1, d_u2, n, p, G);
        cudaDeviceSynchronize();

        // interior and side element results to CPU
        gpuErrorCheck(cudaMemcpy(u, d_u, SIZE, cudaMemcpyDeviceToHost));
        gpuErrorCheck(cudaMemcpy(u1, d_u1, SIZE, cudaMemcpyDeviceToHost));
        gpuErrorCheck(cudaMemcpy(u2, d_u2, SIZE, cudaMemcpyDeviceToHost));

        // 3. Finally update Corner Elements
        updateCorners(u, G);
        printf("\nIteration %d: u(2,2) = %e\n", iter, u[2 * N + 2 % N]);
        //printGrid(u);

        // u2 <- u1; u1 <- u; 
        memcpy(u2, u1, SIZE);
        memcpy(u1, u, SIZE);
    }
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

    // implements finite element music synthesis on 4 by 4 grid
    synthesizeGPU(T);

    return 0;
}