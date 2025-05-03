## Matrix Multiplication

This repository shows my first simple implementations of a parallelized matrix multiplication.
Since matrix multiplications can be done in parallel, they can be sped up by a lot.

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

In the `input` directory is a small python script to generate two NxN matrices, which can be used as inpu matrices. 
By default they are 2500x2500 matrices.

## Implementation

The default serial implementation of the matrix multiplication is in the `normal` folder. It runs on a single thread and scales very poorly. You need to compile the program and then give the two input CSVs. The program saves the result in a new CSV called `result_normal.csv`.

To run the first test you need to run the following commands, assuming you are in the root folder of the project:

```
cd normal
gcc -o matrix matrix.c
mv matrix ../input/
cd ../input/
chmod +x generate.py
./generate.py
./matrix matrix1.csv matrix2.csv
```

The second implementation uses OpenMP to utilize multiple threads. It is located in the `openmp` folder. It scales much better than the default implementation, because it can use all your CPU cores to do the matrix multiplication. The program saves the result in a new CSV called `result_omp.csv`. Running this code requires certain OpenMP-development-libraries to be installed.

To run the second test you need to run the following commands, assuming you are in the root folder of the project:

```
cd openmp
gcc -fopenmp -o matrix_omp matrix.c
mv matrix_omp ../input/
cd ../input/
chmod +x generate.py
./generate.py
./matrix_omp matrix1.csv matrix2.csv
```

The third implementation uses CUDA and requires an Nvidia GPU to run. It is located in the `cuda` folder. This implementation scales very well, because modern Nvidia GPUs have thousands of CUDA-cores. The program saves the result in a new CSV called `result_cuda.csv`. Running this code requires CUDA and its toolchain to run.

To run the second test you need to run the following commands, assuming you are in the root folder of the project:

```
cd cuda
nvcc -o matrix_cuda matrix.cu
mv matrix_cuda ../input/
cd ../input/
chmod +x generate.py
./generate.py
./matrix_cuda matrix1.csv matrix2.csv
```

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
