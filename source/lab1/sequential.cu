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

#define LINE_SIZE 16

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
 * Computes the logic gate result for a given list of
 * tasks consisting of two binary inputs, stores in place.
 * @param tasks an array of pointers to Task structs
 * @param input_length the number of tasks to compute
 */
void sequential(Task* tasks, int input_length) {
    for (int i = 0; i < input_length; i++) {
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
    // ./sequential <input_file_path> <input_file_length> <output_file_path> 
    // ============

    if (argc != 4) {
        printf("\nIncorrect arguments. Run the program as follows:\n");
        printf("./sequential <input_file_path> <input_file_length> <output_file_path>\n");
    }

    int n_inputs;
    sscanf(argv[2], "%d", &n_inputs);

    // Allocate an array for our Tasks
    Task* tasks = (Task*) malloc(n_inputs * sizeof(Task)); // returns pointer to a task

    // Populate the array in-place
    readInput(argv[1], n_inputs, tasks);

    // simulate logic gates sequentially     
    clock_t startTime = clock();
    sequential(tasks, n_inputs);
    clock_t stopTime = clock();
    // CLOCKS_PER_SEC is the number of clock tics elapsed in a second, 
    // which gives us the precise CPU time consumed by a task
    double totalTimeInMs = (((double)stopTime - (double)startTime) * 1000.0) / CLOCKS_PER_SEC;
    // print total time elapsed for sequential gate simulation
    printf("Sequential Simultation: %f ms\n", totalTimeInMs);
    
    writeOutput(argv[3], n_inputs, tasks);
    
    free(tasks);

    return 0;
}