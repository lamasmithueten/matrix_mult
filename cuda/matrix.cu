#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 2500

__global__ void matrixMul(int *A, int *B, int *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    int temp = 0;
    for (int k = 0; k < N; k++) {
      temp += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = temp;
  }
}

void matrixMulWrapper(int *d_A, int *d_B, int *d_C, int N) {
  int blockSize = 16;

  dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
               (N + dimBlock.y - 1) / dimBlock.y);

  matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
}

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

void freeMatrix(int **matrix, int rows) {
  for (int i = 0; i < rows; ++i) {
    free(matrix[i]);
  }
  free(matrix);
}

void copyMatrixToDevice(int **hostMatrix, int **deviceMatrix, int N) {

  cudaMalloc((void **)deviceMatrix, N * N * sizeof(int));

  int *hostMatrixFlat = (int *)malloc(sizeof(int) * N * N);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      hostMatrixFlat[i * N + j] = hostMatrix[i][j];
    }
  }

  cudaMemcpy(*deviceMatrix, hostMatrixFlat, N * N * sizeof(int),
             cudaMemcpyHostToDevice);

  free(hostMatrixFlat);
}

void copyMatrixToHost(int *deviceMatrix, int **hostMatrix, int N) {

  int *hostMatrixFlat = (int *)malloc(sizeof(int) * N * N);

  cudaMemcpy(hostMatrixFlat, deviceMatrix, N * N * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      hostMatrix[i][j] = hostMatrixFlat[i * N + j];
    }
  }

  free(hostMatrixFlat);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix1.csv> <matrix2.csv>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int N = SIZE;

  int **matrixA = allocateMatrix(SIZE, SIZE);
  int **matrixB = allocateMatrix(SIZE, SIZE);
  int **result = allocateMatrix(SIZE, SIZE);
  int *d_matrixA, *d_matrixB, *d_result;

  readMatrixFromCSV(argv[1], matrixA);
  readMatrixFromCSV(argv[2], matrixB);

  copyMatrixToDevice(matrixA, &d_matrixA, SIZE);
  copyMatrixToDevice(matrixB, &d_matrixB, SIZE);
  copyMatrixToDevice(result, &d_result, SIZE);

  matrixMulWrapper(d_matrixA, d_matrixB, d_result, N);

  cudaDeviceSynchronize();

  copyMatrixToHost(d_matrixA, matrixA, SIZE);
  copyMatrixToHost(d_matrixB, matrixB, SIZE);
  copyMatrixToHost(d_result, result, SIZE);

  writeMatrixToCSV("result_cuda.csv", result);

  freeMatrix(matrixA, SIZE);
  freeMatrix(matrixB, SIZE);
  freeMatrix(result, SIZE);

  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_result);

  return EXIT_SUCCESS;
}
