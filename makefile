CC            = gcc
NVCC          = nvcc

CFLAGS_NORMAL = -O3 -march=native -mtune=native -ffast-math -flto
CFLAGS_OMP    = -O3 -march=native -mtune=native -ffast-math -flto -fopenmp
CFLAGS_NVCC   = -O3 -Wno-deprecated-gpu-targets -use_fast_math -Xcompiler -march=native
CFLAGS_MATRIX = -O3 -march=native -mtune=native

OUTPUT_NORMAL = matrix
OUTPUT_OMP    = matrix_omp
OUTPUT_NVCC   = matrix_cuda
OUTPUT_MATRIX = gen_matrix

SRC_NORMAL    = normal/matrix.c normal/header/csv.c
SRC_OMP       = openmp/matrix.c openmp/header/csv.c
SRC_NVCC      = cuda/matrix.cu cuda/header/csv.cu
SRC_MATRIX    = input/input.c 


INPUT1        = matrix1.csv
INPUT2        = matrix2.csv

MATRIX_SIZE   = 5000

#all: set_size $(OUTPUT_MATRIX) $(OUTPUT_NORMAL) $(OUTPUT_OMP) $(OUTPUT_NVCC) create_matrix
all: set_size $(OUTPUT_MATRIX) $(OUTPUT_NORMAL) $(OUTPUT_OMP) create_matrix

create_matrix: 
	[[ -f $(INPUT1) && -f $(INPUT2) ]] && printf "Files exist\n" || ./$(OUTPUT_MATRIX) $(INPUT1) $(INPUT2)

$(OUTPUT_NORMAL): $(SRC_NORMAL)
	$(CC) $(CFLAGS_NORMAL) -o $(OUTPUT_NORMAL) $(SRC_NORMAL)

$(OUTPUT_OMP): $(SRC_OMP)
	$(CC) $(CFLAGS_OMP) -o $(OUTPUT_OMP) $(SRC_OMP)

$(OUTPUT_NVCC): $(SRC_NVCC) 
	$(NVCC) $(CFLAGS_NVCC) -o $(OUTPUT_NVCC) $(SRC_NVCC) 

$(OUTPUT_MATRIX): $(SRC_MATRIX) 
	$(CC) $(CFLAGS_MATRIX) -o $(OUTPUT_MATRIX) $(SRC_MATRIX) 

set_size: 
	find . -name "*.h" -exec sed -i -E 's/SIZE [0-9]+/SIZE $(MATRIX_SIZE)/g' {} \;
	find . -name "*.c" -exec sed -i -E 's/SIZE [0-9]+/SIZE $(MATRIX_SIZE)/g' {} \;
	find . -name "*.cu" -exec sed -i -E 's/SIZE [0-9]+/SIZE $(MATRIX_SIZE)/g' {} \;


clean:
	rm -rf $(OUTPUT_NORMAL) $(OUTPUT_OMP) $(OUTPUT_NVCC) $(INPUT1) $(INPUT2) $(OUTPUT_MATRIX) *.csv

time:
	bash -c	"time ./$(OUTPUT_NORMAL) $(INPUT1) $(INPUT2) "
	bash -c "time ./$(OUTPUT_OMP)  $(INPUT1) $(INPUT2)"
	bash -c "time ./$(OUTPUT_NVCC) $(INPUT1) $(INPUT2)"

pretty:	
	find . -name "*.h" -exec clang-format -i {} \;
	find . -name "*.c" -exec clang-format -i {} \;
	find . -name "*.cu" -exec clang-format -i {} \;
