#include "header/csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMul(int *A, int *B, int *C, int aRows, int bRows,
                          int bCols) {
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < aRows && col < bCols) {
    int temp = 0;
    for (int k = 0; k < bRows; k++) {
      temp += A[row * bRows + k] * B[k * bCols + col];
    }
    C[row * bCols + col] = temp;
  }
}

void matrixMulWrapper(int *d_A, int *d_B, int *d_C, Matrix *A, Matrix *B) {
  dim3 blockSize(32, 32);
  int aRows = A->rows;
  int bRows = B->rows;
  int bCols = B->cols;

  dim3 gridSize((aRows + blockSize.x - 1) / blockSize.x,
                (bCols + blockSize.y - 1) / blockSize.y);

  matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, aRows, bRows, bCols);
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

void copyMatrixToDevice(Matrix *hostMatrix, int **deviceMatrix) {
  cudaMalloc((void **)deviceMatrix,
             hostMatrix->rows * hostMatrix->cols * sizeof(int));

  int *hostMatrixFlat =
      (int *)malloc(sizeof(int) * hostMatrix->rows * hostMatrix->cols);

  for (int i = 0; i < hostMatrix->rows; ++i) {
    for (int j = 0; j < hostMatrix->cols; ++j) {
      hostMatrixFlat[i * hostMatrix->cols + j] = hostMatrix->matrix[i][j];
    }
  }
  cudaMemcpy(*deviceMatrix, hostMatrixFlat,
             hostMatrix->rows * hostMatrix->cols * sizeof(int),
             cudaMemcpyHostToDevice);

  free(hostMatrixFlat);
}

void copyMatrixToHost(int *deviceMatrix, Matrix *hostMatrix) {
  int *hostMatrixFlat =
      (int *)malloc(sizeof(int) * hostMatrix->rows * hostMatrix->cols);

  cudaMemcpy(hostMatrixFlat, deviceMatrix,
             hostMatrix->rows * hostMatrix->cols * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < hostMatrix->rows; ++i) {
    for (int j = 0; j < hostMatrix->cols; ++j) {
      hostMatrix->matrix[i][j] = hostMatrixFlat[i * hostMatrix->cols + j];
    }
  }

  free(hostMatrixFlat);
}

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    fprintf(stderr,
            "Usage: %s <matrix1.csv> <matrix2.csv> (optionally <result.csv>)\n",
            argv[0]);
    return EXIT_FAILURE;
  }
  Matrix matrixA, matrixB, result;
  int *d_matrixA, *d_matrixB, *d_result;

  calcMatrixSize(argv[1], &matrixA);
  calcMatrixSize(argv[2], &matrixB);
  result.rows = matrixA.rows;
  result.cols = matrixB.cols;

  matrixA.matrix = allocateMatrix(matrixA.rows, matrixA.cols);
  matrixB.matrix = allocateMatrix(matrixB.rows, matrixB.cols);
  result.matrix = allocateMatrix(result.rows, result.cols);

  readMatrixFromCSV(argv[1], &matrixA);
  readMatrixFromCSV(argv[2], &matrixB);

  copyMatrixToDevice(&matrixA, &d_matrixA);
  copyMatrixToDevice(&matrixB, &d_matrixB);
  copyMatrixToDevice(&result, &d_result);

  matrixMulWrapper(d_matrixA, d_matrixB, d_result, &matrixA, &matrixB);

  cudaDeviceSynchronize();
  copyMatrixToHost(d_matrixA, &matrixA);
  copyMatrixToHost(d_matrixB, &matrixB);
  copyMatrixToHost(d_result, &result);

  if (argc == 3) {
    writeMatrixToCSV("result_cuda.csv", &result);
  } else {
    writeMatrixToCSV(argv[3], &result);
  }

  freeMatrix(matrixA.matrix, matrixA.rows);
  freeMatrix(matrixB.matrix, matrixB.rows);
  freeMatrix(result.matrix, result.rows);

  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_result);

  return EXIT_SUCCESS;
}
