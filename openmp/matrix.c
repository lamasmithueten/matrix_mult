#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 2500

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

void matrixMultiply(int **A, int **B, int **C) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      C[i][j] = 0;
      for (int k = 0; k < SIZE; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
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

int **allocateMatrix(int rows, int cols) {
  int **matrix = (int **)malloc(rows * sizeof(int *));
  if (matrix == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }


#pragma omp parallel for 
  for (int i = 0; i < rows; ++i) {
    matrix[i] = (int *)malloc(cols * sizeof(int));
    if (matrix[i] == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
  }

  return matrix;
}

void freeMatrix(int **matrix, int rows) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    free(matrix[i]);
  }
  free(matrix);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix1.csv> <matrix2.csv>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int **matrixA = allocateMatrix(SIZE, SIZE);
  int **matrixB = allocateMatrix(SIZE, SIZE);
  int **result = allocateMatrix(SIZE, SIZE);

  readMatrixFromCSV(argv[1], matrixA);
  readMatrixFromCSV(argv[2], matrixB);

  matrixMultiply(matrixA, matrixB, result);

  writeMatrixToCSV("result_omp.csv", result);

  freeMatrix(matrixA, SIZE);
  freeMatrix(matrixB, SIZE);
  freeMatrix(result, SIZE);

  return EXIT_SUCCESS;
}
