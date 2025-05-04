CC            = gcc
NVCC          = nvcc

CFLAGS_NORMAL = -O3 -march=native -mtune=native 
CFLAGS_OMP    = -O3 -march=native -mtune=native -fopenmp
CFLAGS_NVCC   = -O3 -Wno-deprecated-gpu-targets

OUTPUT_NORMAL = matrix
OUTPUT_OMP    = matrix_omp
OUTPUT_NVCC   = matrix_cuda

SRC_NORMAL    = normal/matrix.c normal/header/csv.c
SRC_OMP       = openmp/matrix.c openmp/header/csv.c
SRC_NVCC      = cuda/matrix.cu cuda/header/csv.cu


INPUT1        = matrix1.csv
INPUT2        = matrix2.csv
PYTHON_SCRIPT = input/generate.py

all: $(OUTPUT_NORMAL) $(OUTPUT_OMP) $(OUTPUT_NVCC) create_matrix

create_matrix: $(OUTPUT_NORMAL) $(OUTPUT_OMP) $(OUTPUT_NVCC)
	python3 $(PYTHON_SCRIPT)

$(OUTPUT_NORMAL): $(SRC_NORMAL)
	$(CC) $(CFLAGS_NORMAL) -o $(OUTPUT_NORMAL) $(SRC_NORMAL)

$(OUTPUT_OMP): $(SRC_OMP)
	$(CC) $(CFLAGS_OMP) -o $(OUTPUT_OMP) $(SRC_OMP)

$(OUTPUT_NVCC): $(SRC_NVCC) 
	$(NVCC) $(CFLAGS_NVCC) -o $(OUTPUT_NVCC) $(SRC_NVCC) 


clean:
	rm -rf $(OUTPUT_NORMAL) $(OUTPUT_OMP) $(OUTPUT_NVCC) $(INPUT1) $(INPUT2) 
