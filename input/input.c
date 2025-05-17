#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 5000

void createMatrix(const char *filename, int rows, int cols) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      fprintf(file, "%d", rand() % 99 + 1);
      if (j < cols - 1) {
        fprintf(file, ",");
      }
    }
    fprintf(file, "\n");
  }

  fclose(file);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix1.csv> <matrix2.csv\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  srand(time(0));

  int range = SIZE / 10;
  int n = SIZE + (rand() % range + (-range / 2));
  int m = SIZE + (rand() % range + (-range / 2));
  int p = SIZE + (rand() % range + (-range / 2));

  createMatrix(argv[1], n, m);
  createMatrix(argv[2], m, p);

  return EXIT_SUCCESS;
}
