%%writefile global_queuing.cu
#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_QUEUE_SIZE 200000
// #define BLOCK_SIZE 32 // blockSize (32, 64, 128)
// #define NUM_BLOCKS 10 // numBlock (10, 25, 35)

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

__device__ int numNextLevelNodes_d = 0;

// Macro for Cuda Debugging
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__ int gate_solver(int gate, int input1, int input2) {
    int output;
    switch (gate) {
            case AND:
                output = input1 && input2;
                break;
            case OR:
                output = input1 || input2;
                break;
            case NAND:
                output = !(input1 && input2);
                break;
            case NOR:
                output = !(input1 || input2);
                break;
            case XOR:
                output = (input1 || input2) && !(input1 && input2);
                break;
            case XNOR:
                output = (input1 && input2) || (!input1 && !input2);
                break;
    }
    return output;
}

/**
 * @brief Reads an input file containing length L in first line and one integer for each of the next L lines, storing the values in the integer array provided.
 * 
 * @param input1 pointer to an integer array to store the data
 * @param filepath character array, name of input file to open
 * @return int, length of resulting array
 */
int read_input_one_two_four(int **input1, char* filepath){
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    }
    
    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    
    // Allocate len entries and assign the pointer to out input reference
    *input1 = ( int *)malloc(len * sizeof(int));
    int temp1;
    
    // read line by line, one integer per line
    while (fscanf(fp, "%d", &temp1) == 1) {
        (*input1)[counter] = temp1;

        counter++;
    }

    fclose(fp);
    return len;

}

/**
 * @brief Reads a CSV formatted file with information for every node gate.
 * 
 * @param input1 Pointer to Visited array
 * @param input2 Pointer to Logic Gate types array
 * @param input3 Pointer to Logic Gate inputs array
 * @param input4 Pointer to Logic Gate outputs array
 * @param filepath 
 * @return int, the number of node entries
 */
int read_input_three(int** input1, int** input2, int** input3, int** input4,char* filepath){
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    }
    
    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = ( int *)malloc(len * sizeof(int));    // visited[i], either 0 or 1
    *input2 = ( int *)malloc(len * sizeof(int));    // gate[i], see macros definitions
    *input3 = ( int *)malloc(len * sizeof(int));    // input[i], fixed input to node i
    *input4 = ( int *)malloc(len * sizeof(int));    // output[i], -1 if not computed

    int temp1;
    int temp2;
    int temp3;
    int temp4;
    while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4) {
        (*input1)[counter] = temp1;
        (*input2)[counter] = temp2;
        (*input3)[counter] = temp3;
        (*input4)[counter] = temp4;
        counter++;
    }

    fclose(fp);
    return len; // number of total nodes in circuit
}


__global__ void globalQueueBFS(int* currLevelNodes, int numCurrLevelNodes, int* nodePtrs, int* nodeNeighbors, int* nodeVisited, int* nodeGate, int* nodeInput, int* nodeOutput, int* nextLevelNodes, int nodes_per_thread) {
    
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int start = tid * nodes_per_thread;
    const int thread_limit = start + nodes_per_thread;
    const bool debug = tid == 0; 

    for (unsigned int idx = start; idx < thread_limit && idx < numCurrLevelNodes; idx++) {
        // get the node
        unsigned int node = currLevelNodes[idx];

        // Loop through neighbors
        for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; nbrIdx++) {
            int neighbor = nodeNeighbors[nbrIdx];
            // If neighbor hasn't been visited yet
            int already_visited = atomicExch(&(nodeVisited[neighbor]), 1);
            if (!already_visited) {
                int output = gate_solver(nodeGate[neighbor], nodeInput[neighbor], nodeOutput[node]);
                nodeOutput[neighbor] = output;
                // Atomically add to queue
                int new_i = atomicAdd(&numNextLevelNodes_d, 1);
                nextLevelNodes[new_i] = neighbor;

                // if (debug) {
                //     printf("Result nextQueue[%d]=%d\n", new_i, nextLevelNodes[new_i]);
                // }
            }
        }
    }
    // printf("Thread %d finished processing queue %d-%d\n", tid, start, start + thread_limit);

}


int main(int argc, char *argv[]) {
    // Variables
    if (argc != 9)
    {
        printf("\nIncorrect number of arguments. Please run the program as follows:\n");
        printf("./global_queuing <numBlock> <blockSize> <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath>\n");
    }
    // Read input and output file names from command
    int numBlock;
    int blockSize; 
    sscanf(argv[1], "%d", &numBlock);
    sscanf(argv[2], "%d", &blockSize);

    char* input_1 = argv[3]; 
    char* input_2 = argv[4]; 
    char* input_3 = argv[5];
    char* input_4 = argv[6];
    char* output_nodeOutput = argv[7]; 
    char* output_nextLevelNodes = argv[8];
    
    // Variables
    int numNodePtrs;          
    int numNodes;             
    int *nodePtrs_h;          
    int *nodeNeighbors_h;      
    int *nodeVisited_h;   
    int numTotalNeighbors_h;
    int *currLevelNodes_h;
    int numCurrLevelNodes;
    int numNextLevelNodes_h;
    int *nodeGate_h;
    int *nodeInput_h;
    int *nodeOutput_h;
    
    numNextLevelNodes_h = 0;
    numNodePtrs = read_input_one_two_four(&nodePtrs_h, input_1);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, input_2);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, input_3);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input_4);
    
    // Allocate GPU Memory 

    int *nodePtrs_d;          
    int *nodeNeighbors_d;      
    int *nodeVisited_d;   
    int numTotalNeighbors_d;
    int *currLevelNodes_d;
    int *nodeGate_d;
    int *nodeInput_d;
    int *nodeOutput_d;

    int* nextLevelNodes_d;
    gpuErrorCheck(cudaMalloc((void **)&nextLevelNodes_d, MAX_QUEUE_SIZE * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void **)&nodePtrs_d, numNodePtrs * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void **)&nodeNeighbors_d, numTotalNeighbors_h * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void **)&currLevelNodes_d, numCurrLevelNodes * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void **)&nodeGate_d, numNodes * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void **)&nodeInput_d, numNodes * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void **)&nodeOutput_d, numNodes * sizeof(int)));

    gpuErrorCheck(cudaMemcpy(nodeVisited_d, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(nodePtrs_d, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(currLevelNodes_d, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(nodeGate_d, nodeGate_h, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(nodeInput_d, nodeInput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(nodeOutput_d, nodeOutput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice));


    // Start timer
    float memsettime;
    cudaEvent_t start, stop;
    gpuErrorCheck(cudaEventCreate(&start));
    gpuErrorCheck(cudaEventCreate(&stop));
    gpuErrorCheck(cudaEventRecord(start, 0));

    // CALL THE KERNEL
    printf("%d Blocks, %d Threads/Block\n", numBlock, blockSize);

    int capacity = numBlock * blockSize;
    int remainder = (int)((numCurrLevelNodes % capacity) > 0);
    int nodes_per_thread = (numCurrLevelNodes / capacity) + remainder;
    printf("%d Nodes per Thread\n", nodes_per_thread);
    globalQueueBFS<<<numBlock, blockSize>>>(currLevelNodes_d, numCurrLevelNodes, nodePtrs_d, nodeNeighbors_d, nodeVisited_d, nodeGate_d, nodeInput_d, nodeOutput_d, nextLevelNodes_d, nodes_per_thread);
    cudaDeviceSynchronize();

    // Stop timer
    gpuErrorCheck(cudaEventRecord(stop, 0));
    gpuErrorCheck(cudaEventSynchronize(stop));
    gpuErrorCheck(cudaEventElapsedTime(&memsettime, start, stop));
    // print total time elapsed for global queuing gate simulation
    printf("*** Global Queuing Execution Time: %f ms***\n", memsettime);

    gpuErrorCheck(cudaEventDestroy(start));
    gpuErrorCheck(cudaEventDestroy(stop));

    // Cleanup;
    cudaFree(nodePtrs_d);
    free(nodePtrs_h);
    cudaFree(nodeNeighbors_d);
    free(nodeNeighbors_h);
    cudaFree(nodeVisited_d);
    free(nodeVisited_h);
    cudaFree(currLevelNodes_d);
    free(currLevelNodes_h);
    cudaFree(nodeGate_d);
    free(nodeGate_h);
    cudaFree(nodeInput_d);
    free(nodeInput_h);

    // Copy results
    gpuErrorCheck(cudaMemcpyFromSymbol(&numNextLevelNodes_h, numNextLevelNodes_d, sizeof(int)));
    int *nodeOutputResult;
    nodeOutputResult = (int*)malloc(numNodes*sizeof(int)); 
    gpuErrorCheck(cudaMemcpy(nodeOutputResult, nodeOutput_d, numNodes*sizeof(int), cudaMemcpyDeviceToHost));

    int *nextLevelNodes_h; // to write in output file
    nextLevelNodes_h = ( int *)malloc(numNextLevelNodes_h * sizeof(int));

    gpuErrorCheck(cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d, numNextLevelNodes_h * sizeof(int), cudaMemcpyDeviceToHost));

    FILE* output_file_node = fopen(output_nodeOutput, "w");
    FILE* output_file_nextLevelNode = fopen(output_nextLevelNodes, "w");

    // Print out the results
    fprintf(output_file_node, "%d\n", numNodes);
    for (int i = 0; i < numNodes; i++) { 
        fprintf(output_file_node, "%d\n", nodeOutputResult[i]); 
    }
    fclose(output_file_node);

    fprintf(output_file_nextLevelNode, "%d\n", numNextLevelNodes_h);
    for (int i = 0; i < numNextLevelNodes_h; i++) { 
        fprintf(output_file_nextLevelNode, "%d\n", nextLevelNodes_h[i]); 
    }
    fclose(output_file_nextLevelNode);
}