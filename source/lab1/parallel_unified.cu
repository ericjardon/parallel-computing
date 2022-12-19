%%writefile parallel_unified.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__
#endif

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

#define LINE_SIZE 16
#define THREADS_PER_BLOCK 1024

// Macro for Cuda Debugging
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Container for our working data, single Task
typedef struct Task {
    char gate;
    char input1;
    char input2;
    char output;
} Task;

/**
 * @brief Read the given filename, populate inputs 1 and 2
 * of our Task pointer array. Assumes each line 
 * to be LINE_SIZE characters long.
 * 
 * @param input_file String name of the file to read
 * @param length Number of lines in the file
 * @param tasks Pointer to our Task pointer array
 */
void readInput(char* input_file, int length, Task* tasks) {
    FILE* f = fopen(input_file, "r");

    char line[LINE_SIZE]; // char array length 16
    int i = 0;

    while (fgets(line, LINE_SIZE, f)) {
        if (i >= length) break;
        // convert to ints with - '0'
        tasks[i].input1 = line[0] - '0';
        tasks[i].input2 = line[2] - '0';
        tasks[i].gate = line[4] - '0';
        i++;
    }

    fclose(f);
}

/**
 * @brief Produces the output file containing a single column of
 * all the logic gate outputs from the tasks array.
 * 
 * @param output_file Name of the file to write.
 * @param length Number of tasks
 * @param tasks Pointer to our Task pointer array
 */
void writeOutput(char* output_file, int length, Task* tasks) {
    FILE* f = fopen(output_file, "w+"); // creates the file if not exists

    for (int i = 0; i < length; i++) {
        // Adding + '0' to convert into char
        fputc(tasks[i].output + '0', f);
        fputc('\n', f);
    }

    fclose(f);
}

/**
 * @brief 
 * Computes the logic gate result for a given task
 * consisting of two binary inputs, stores in place.
 * @param tasks an array of pointers to Task structs
 */
__global__ void parallel_unified(Task* tasks, int n_inputs) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n_inputs) {
        switch (tasks[i].gate) {
                case AND:
                    tasks[i].output = tasks[i].input1 && tasks[i].input2;
                    break;
                case OR:
                    tasks[i].output = tasks[i].input1 || tasks[i].input2;
                    break;
                case NAND:
                    tasks[i].output = !(tasks[i].input1 && tasks[i].input2);
                    break;
                case NOR:
                    tasks[i].output = !(tasks[i].input1 || tasks[i].input2);
                    break;
                case XOR:
                    tasks[i].output = (tasks[i].input1 || tasks[i].input2) && !(tasks[i].input1 && tasks[i].input2);
                    break;
                case XNOR:
                    tasks[i].output = (tasks[i].input1 && tasks[i].input2) || (!tasks[i].input1 && !tasks[i].input2);
                    break;
            }
    }
}

int main(int argc, char *argv[])
{
    // ============
    // RUN COMMAND:
    // ./parallel_unified <input_file_path> <input_file_length> <output_file_path> 
    // ============

    if (argc != 4) {
        printf("\nIncorrect arguments. Run the program as follows:\n");
        printf("./parallel_unified <input_file_path> <input_file_length> <output_file_path>\n");
    }

    int n_inputs;
    sscanf(argv[2], "%d", &n_inputs); // n_threads

    size_t TASKS_SIZE = n_inputs * sizeof(Task);
    // Allocate an array for our Tasks, accesible from both CPU and GPU
    Task* tasks;
    gpuErrchk(cudaMallocManaged((void **)&tasks, TASKS_SIZE));
    // Populate the array in-place
    readInput(argv[1], n_inputs, tasks);

    float memsettime;
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, 0));
    
    parallel_unified<<<ceil(n_inputs/THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(tasks, n_inputs); // n threads for n tasks
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop);
    printf("Parallel w/ Unified Memory Allocation: %f ms\n", memsettime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    writeOutput(argv[3], n_inputs, tasks);

    cudaFree(tasks);
    return 0;
}