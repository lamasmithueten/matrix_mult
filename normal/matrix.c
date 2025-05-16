#include "header/config.h"
#include "header/csv.h"
#include <stdio.h>
#include <stdlib.h>


void matrixMultiply(Matrix * A, Matrix * B, Matrix * C) {
  for (int i = 0; i < A->rows; ++i) {
    for (int j = 0; j < B->cols; ++j) {
      C->matrix[i][j] = 0;
      for (int k = 0; k < B->rows; ++k) {
        C->matrix[i][j] += A->matrix[i][k] * B->matrix[k][j];
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

Matrix matrixA, matrixB, result;

calcMatrixSize(argv[1], &matrixA);
calcMatrixSize(argv[2], &matrixB);
result.rows=matrixA.rows;
result.cols=matrixB.cols;


  matrixA.matrix = allocateMatrix(matrixA.rows, matrixA.cols);
  matrixB.matrix = allocateMatrix(matrixB.rows, matrixB.cols);
  result.matrix = allocateMatrix(result.rows, result.cols);

  readMatrixFromCSV(argv[1], &matrixA);
  readMatrixFromCSV(argv[2], &matrixB);

  matrixMultiply(&matrixA, &matrixB, &result);

  writeMatrixToCSV("result_normal.csv", &result);

  freeMatrix(matrixA.matrix, matrixA.rows );
  freeMatrix(matrixB.matrix, matrixB.rows);
  freeMatrix(result.matrix, result.rows);

  return EXIT_SUCCESS;
}
