%%writefile sequential.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

#define MAX_QUEUE_SIZE 200000


int gate_solver(int gate, int input1, int input2) {
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

void BFS(int* currLevelNodes, int numCurrLevelNodes, int* nodePtrs, int* nodeNeighbors, int* nodeVisited, int* nodeGate, int* nodeInput, int* nodeOutput, int* nextLevelNodes, int* numNextLevelNodes) {
    // Traverse current level queue
    for (int idx = 0; idx < numCurrLevelNodes; idx++) { // while idx < numCurrLevelNodes
        int node = currLevelNodes[idx];
        // Loop over all neighbors of the node
        // printf("Nodeptrs+1 %d \n", nodePtrs[node + 1]);
        for (int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; nbrIdx++) {
            int neighbor = nodeNeighbors[nbrIdx];
            // If neighbor hasn't been visited yet
            if (!nodeVisited[neighbor]) {
                // Mark it as visited and add it to queue
                nodeVisited[neighbor] = 1;
                nodeOutput[neighbor] = gate_solver(nodeGate[neighbor], nodeInput[neighbor], nodeOutput[node]);
                // printf("nextLevelNodes LEN %d index %d", numNextLevelNodes);
                nextLevelNodes[*numNextLevelNodes] = neighbor;
                ++(*numNextLevelNodes);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 7)
    {
        printf("\nIncorrect number of arguments. Please run the program as follows:\n");
        printf("./sequential <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath>\n");
    }
    
    // Read input and output file names from command
    char* input_1 = argv[1]; 
    char* input_2 = argv[2]; 
    char* input_3 = argv[3]; 
    char* input_4 = argv[4];
    char* output_nodeOutput = argv[5]; 
    char* output_nextLevelNodes = argv[6];
    
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

    int *nextLevelNodes_h; // to write in output file
    nextLevelNodes_h = ( int *)malloc(MAX_QUEUE_SIZE * sizeof(int));
    
    numNextLevelNodes_h = 0;

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, input_1);

    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, input_2);

    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, input_3);

    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input_4);
    // Measure runtime of sequential BFS
    clock_t startTime = clock();
    BFS(currLevelNodes_h, numCurrLevelNodes, nodePtrs_h, nodeNeighbors_h, nodeVisited_h, nodeGate_h, nodeInput_h, nodeOutput_h, nextLevelNodes_h, &numNextLevelNodes_h);
    clock_t stopTime = clock();

    // CLOCKS_PER_SEC is the number of clock tics elapsed in a second, 
    // which gives us the precise CPU time consumed by a task
    double totalTimeInMs = (((double)stopTime - (double)startTime) * 1000.0) / CLOCKS_PER_SEC;
    // print total time elapsed for sequential gate simulation
    printf("Sequential Simultation: %f ms\n", totalTimeInMs);

    FILE* output_file_node = fopen(output_nodeOutput, "w"); 
    FILE* output_file_nextLevelNode = fopen(output_nextLevelNodes, "w");

    // Print out the results
    fprintf(output_file_node, "%d\n", numNodes);
    for (int i = 0; i < numNodes; i++) { 
        fprintf(output_file_node, "%d\n", nodeOutput_h[i]); 
    }
    fclose(output_file_node);

    fprintf(output_file_nextLevelNode, "%d\n", numNextLevelNodes_h);
    for (int i = 0; i < numNextLevelNodes_h; i++) { 
        fprintf(output_file_nextLevelNode, "%d\n", nextLevelNodes_h[i]); 
    }
    fclose(output_file_nextLevelNode);
}
