#include <stdio.h>
#include <stdlib.h>
void sort(int *pointer, int size){
  //get from https://stackoverflow.com/questions/13012594/sorting-with-pointers-instead-of-indexes
    int *i, *j, temp;
    for(i = pointer; i < pointer + size; i++){
        for(j = i + 1; j < pointer + size; j++){
            if(*j < *i){
                temp = *j;
                *j = *i;
                *i = temp;
            }
        }
    }
}

void compareNextLevelNodeFiles(char *file_name1, char *file_name2) 
{ 

  
    FILE* fp_1 = fopen(file_name1, "r");
    if (fp_1 == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    } 

    FILE* fp_2 = fopen(file_name2, "r");
    if (fp_2 == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    } 
    
    int counter = 0;
    int len_1;
    int len_2;
    int length_file_1 = fscanf(fp_1, "%d", &len_1);
    int length_file_2 = fscanf(fp_2, "%d", &len_2);

    if(length_file_1 != length_file_2){
      fprintf(stderr, "Wrong file lengths: f1 %d f2 %d\n", len_1, len_2);
      exit(1);
    }
    int *input1 = (int *)malloc(len_1 * sizeof(int));
    int *input2 = (int *)malloc(len_2 * sizeof(int));




    int temp1;
    int temp2;

    while ((fscanf(fp_1, "%d", &temp1) == 1) && (fscanf(fp_2, "%d", &temp2) == 1)) {
        (input1)[counter] = temp1;
        (input2)[counter] = temp2;
        counter++;
    }

    sort(input1, len_1);
    sort(input2, len_2);

    for(int i=0; i< len_1; i++){
      if(input1[i] != input2[i]){
        fprintf(stderr, "Something goes wrong\n");
        exit(1);
      }
    }

    fprintf(stderr, "No errors!\n");
        exit(1);

} 

int main(int argc, char *argv[]){

    if( argc < 3) {
      printf("Require two files\n");
      exit(1);
      
   }
compareNextLevelNodeFiles(argv[1], argv[2]);
}
