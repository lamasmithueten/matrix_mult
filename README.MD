## Matrix Multiplication

This repository shows my first simple implementations of a parallelized matrix multiplication.
It is my first atempt to get a feeling for programming with parallel programming libraries and their toolchains. It is by no means a professional implementation and still very basic.

Since matrix multiplications can be done in parallel, they can be sped up by using multiple CPU or GPU cores. Additionally compiler flags for auto-vectorization have been enabled in the makefile.

The repository has the following structure:

```
.
├── cuda
│   └── matrix.cu
├── input
│   └── generate.py
├── normal
│   └── matrix.c
└── openmp
    └── matrix.c
```

Each C-file takes two matrices as input. They are stored in CSV files.

In the `input` directory is a small python script to generate two NxN matrices, which can be used as input matrices. 
By default they are 2500x2500 matrices.

## Implementation

To show how parallelization can increase performance, I have implemented matrix multiplication in 3 ways. Using plain single threaded C, C in combination with OpenMP to use multiple CPU cores and using C with CUDA to utilize the CUDA cores of an Nvidia GPU.

To compile the application you can use the `makefile` and compile all three programs using the `make` command in the root of the directory. It creates 2 input CSVs and three binaries `matrix`, `matrix_omp` and `matrix_cuda`. Each binary requires two input CSV and creates a result CSV. I used files as input and output to reuse the same input for different binaries. Also printing large result matrices onto the screen requires a lot CPU-time, which isn't what I am benchmarking.

Adjust the compiler flags or compiler in the makefile to your needs. 


## Results

In the end you should have the same threet result CSVs, but they were calculated at different speeds.

With an Intel Core I7 6700K and an Nvidia RTX 2080 ti I had the following results:

```
$ time ./matrix matrix1.csv matrix2.csv; time ./matrix_omp matrix1.csv matrix2.csv; time ./matrix_cuda matrix1.csv matrix2.csv

real    2m9,997s
user    2m9,433s
sys     0m0,146s

real    0m30,031s
user    3m37,271s
sys     0m0,666s

real    0m1,505s
user    0m1,197s
sys     0m0,290s
```
