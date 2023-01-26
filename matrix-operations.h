#ifndef MATRIX_OPERATIONS
#define MATRIX_OPERATIONS

#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Matrix
{
    int rows;
    int columns;
    DATA_TYPE** values;
} Matrix;

DATA_TYPE** allocateMatrix(int rows, int columns);

Matrix* createMatrix(DATA_TYPE** values, int rows, int columns);

void deleteMatrix(Matrix* matrix);

void fillRandomMatrix(DATA_TYPE** matrix, int rows, int columns, int minimum, int maximum, float zeroProbability);

DATA_TYPE** generateRandomMatrix(int rows, int columns, int minimum, int maximum, float zeroProbability);

Matrix* createRandomMatrix(int rows, int columns, int minimum, int maximum, float zeroProbability);

void printMatrix(Matrix* matrix);

Matrix* standardDotProduct(Matrix* matrix1, Matrix* matrix2);

Matrix* parallelDotProduct(Matrix* matrix1, Matrix* matrix2);

int isEqual(Matrix* matrix1, Matrix* matrix2);

#endif