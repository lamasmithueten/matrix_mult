#include "csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void readMatrixFromCSV(const char *filename, Matrix * matrix) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < matrix->rows; ++i) {
    for (int j = 0; j < matrix->cols; ++j) {
      if (fscanf(file, "%d,", &matrix->matrix[i][j]) != 1) {
        fprintf(stderr, "Error reading file\n");
	printf("Row: %d\nCol: %d\n", i,j);
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}
void writeMatrixToCSV(const char *filename, Matrix * matrix) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < matrix->rows; ++i) {
    for (int j = 0; j < matrix->cols; ++j) {
      fprintf(file, "%d", matrix->matrix[i][j]);
      if (j < matrix->cols - 1) {
        fprintf(file, ",");
      }
    }
    fprintf(file, "\n");
  }

  fclose(file);
}


void calcMatrixSize(const char * filename, Matrix * matrix){
	FILE * file = fopen(filename, "r");
  if (file == NULL) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }
	
    int rows = 0;
    int cols = 0;
    char line[1024] = {0};
int count =1;
while (count){
	size_t limit = fread(line, 1, sizeof(line), file);
	for (int i = 0; i< limit; i++){
		if(line[i] == ',')cols++;
		if(line[i] == '\n'){count--; break;	}
	}
}

rewind (file);

while (1){
	size_t limit = fread(line, 1, sizeof(line), file);
	for (int i = 0; i< limit; i++){
		if(line[i] == '\n') rows++;
	}
	if(feof(file)) break;
}


	


	matrix->rows=rows;
	matrix->cols=++cols;
	
	fclose(file);
}
