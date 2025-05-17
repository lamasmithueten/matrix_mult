#include "header/csv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void matrixMultiply(Matrix *A, Matrix *B, Matrix *C) {
#pragma omp parallel for
  for (int i = 0; i < A->rows; ++i) {
    for (int j = 0; j < B->cols; ++j) {
    int sum = 0;
            #pragma omp simd reduction(+:sum)
      for (int k = 0; k < B->rows; ++k) {
        sum += A->matrix[i][k] * B->matrix[k][j];
      }
      C->matrix[i][j] = sum;
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
  if (argc < 3 || argc > 4) {
    fprintf(stderr,
            "Usage: %s <matrix1.csv> <matrix2.csv> (optionally <result.csv>)\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  Matrix matrixA, matrixB, result;

  calcMatrixSize(argv[1], &matrixA);
  calcMatrixSize(argv[2], &matrixB);
  result.rows = matrixA.rows;
  result.cols = matrixB.cols;

  matrixA.matrix = allocateMatrix(matrixA.rows, matrixA.cols);
  matrixB.matrix = allocateMatrix(matrixB.rows, matrixB.cols);
  result.matrix = allocateMatrix(result.rows, result.cols);

  readMatrixFromCSV(argv[1], &matrixA);
  readMatrixFromCSV(argv[2], &matrixB);

  matrixMultiply(&matrixA, &matrixB, &result);

  if (argc == 3) {
    writeMatrixToCSV("result_omp.csv", &result);
  } else {
    writeMatrixToCSV(argv[3], &result);
  }

  freeMatrix(matrixA.matrix, matrixA.rows);
  freeMatrix(matrixB.matrix, matrixB.rows);
  freeMatrix(result.matrix, result.rows);

  return EXIT_SUCCESS;
}
