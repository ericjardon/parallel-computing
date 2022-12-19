#include <stdio.h>
#include <stdlib.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

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

int main(int argc, char *argv[]) {
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

  numNodePtrs = read_input_one_two_four(&nodePtrs_h, "input1.raw");

  numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, "input2.raw");

  numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h,"input3.raw");

  numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, "input4.raw");
}
