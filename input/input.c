#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000

void createMatrix(const char *filename) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      fprintf(file, "%d", rand() % 99 + 1);
      if (j < SIZE - 1) {
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

  createMatrix(argv[1]);
  createMatrix(argv[2]);

  return EXIT_SUCCESS;
}
