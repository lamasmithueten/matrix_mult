#ifndef CSV_H_
#define CSV_H_

typedef struct{
	int ** matrix;
	int rows, cols;
} Matrix;
void readMatrixFromCSV(const char *filename, Matrix * matrix);
void writeMatrixToCSV(const char *filename, Matrix *matrix);
void calcMatrixSize(const char * filename, Matrix * matrix);

#endif
