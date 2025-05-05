#include "config.h"
#include "csv.h"
#include <stdio.h>
#include <stdlib.h>

void readMatrixFromCSV(const char *filename, int **matrix) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      if (fscanf(file, "%d,", &matrix[i][j]) != 1) {
        fprintf(stderr, "Error reading file\n");
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}
void writeMatrixToCSV(const char *filename, int **matrix) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      fprintf(file, "%d", matrix[i][j]);
      if (j < SIZE - 1) {
        fprintf(file, ",");
      }
    }
    fprintf(file, "\n");
  }

  fclose(file);
}
