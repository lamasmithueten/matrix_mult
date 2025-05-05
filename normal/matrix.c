#include "header/config.h"
#include "header/csv.h"
#include <stdio.h>
#include <stdlib.h>

void matrixMultiply(int **A, int **B, int **C) {
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      C[i][j] = 0;
      for (int k = 0; k < SIZE; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int **allocateMatrix(int rows, int cols) {
  int **matrix = (int **)malloc(rows * sizeof(int *));
  if (matrix == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

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

  writeMatrixToCSV("result_normal.csv", result);

  freeMatrix(matrixA, SIZE);
  freeMatrix(matrixB, SIZE);
  freeMatrix(result, SIZE);

  return EXIT_SUCCESS;
}
