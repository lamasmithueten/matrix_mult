#!/bin/python

import csv
import random

def generate_matrix(filename, size=2500):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(size):
            row = [random.randint(1, 100) for _ in range(size)]
            writer.writerow(row)

# Generate two 100x100 matrices
generate_matrix('matrix1.csv')
generate_matrix('matrix2.csv')

print("CSV files 'matrix1.csv' and 'matrix2.csv' have been generated.")
