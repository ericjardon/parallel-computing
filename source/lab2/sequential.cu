%%writefile sequential.cu
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Implements a 4 by 4 finite element grid sequentially
 *
 * @param N the number of rows and columns of grid
 * @param n_iterations the number of iterations to run simulation
 */
void sequential(int N, int n_iterations) {
    float n = 0.0002;
    float p = 0.5;
    float G = 0.75;

    size_t grid_size = N * N * sizeof(float);
	float* u = (float*)calloc(N*N, grid_size);
	float* u1 = (float*)calloc(N*N, grid_size);
	float* u2 = (float*)calloc(N*N, grid_size);

    // Initialize
    u1[((N * N)/ 2 + N / 2)] = 1;
    
    for (int iter = 0; iter < n_iterations; iter++) {
        
        // Update interior elements
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                u[N * i + j] = ((p * (u1[N * (i-1) + j] + u1[N* (i+1) + j] + u1[N * i + (j-1)] + u1[N * i + (j+1)] - 4 * u1[N * i + j])) + 
                           (2 * u1[N * i + j]) - ((1 - n) * u2[N * i + j])) / (1+n);
            }
        }
       
        // Update side elements
        for (int i = 1; i < N - 1; i++) {
            u[i] = G * u[N + i];
            u[(N * (N-1)) + i] = G * u[(N * (N - 2)) + i];
        }
        
        for (int i = 1; i < N - 1; i++) {
            u[i * N] = G * u[(N * i) + 1];
            u[(i * N) + N - 1] = G * u[(N * i) + (N-2)];
        }

        // Update corner elements
        u[0] = G * u[N];
        u[N - 1] = G * u[N - 2];
        u[N * (N - 1)] = G * u[N * (N - 2)];
        u[(N * (N - 1)) + (N - 1)] = G * u[(N * (N - 1)) + (N - 2)];

        printf("Iteration %d: u(2,2) = %f\n", iter, u[(N * (N/2)) + N/2]);
        
        // Update references
        free(u2);
        u2 = u1;
        u1 = u;
        u = (float*)calloc(N * N, grid_size);
    }

    free(u);
    free(u1);
    free(u2);
}


int main(int argc, char *argv[])
{
    // ============
    // RUN COMMAND:
    // ./sequential <number_of_iterations> 
    // ============

    if (argc != 2) {
        printf("\nIncorrect arguments. Run the program as follows:\n");
        printf("./sequential <number_of_iterations>\n");
    }

    int T;
    int N = 4;
    sscanf(argv[1], "%d", &T);

    // implements 4 by 4 finite grid sequentially    
    sequential(N, T);

    return 0;
}