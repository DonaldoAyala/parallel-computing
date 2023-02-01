#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

typedef struct Matrix
{
    int rows;
    int columns;
    DATA_TYPE** values;
} Matrix;

Matrix* createMatrix(DATA_TYPE** values, int rows, int columns)
{
    Matrix* newMatrix = (Matrix*) malloc(sizeof(Matrix));
    newMatrix -> rows = rows;
    newMatrix -> columns = columns;
    newMatrix -> values = values;

    return newMatrix;
}

DATA_TYPE** allocateMatrix(int rows, int columns)
{
    DATA_TYPE** newMatrix = (DATA_TYPE**) malloc(rows * sizeof(DATA_TYPE*));
    int row = 0, col;
    for (row = 0; row < rows; row++)
    {
        newMatrix[row] = (DATA_TYPE*)malloc(columns * sizeof(DATA_TYPE));
    }

    return newMatrix;
}

void deleteMatrix(Matrix* matrix)
{
    if (matrix == NULL)
    {
        return;
    }

    int row = 0;
    for (row = 0; row < matrix -> rows; row++)
    {
        free(matrix -> values[row]);
    }

    free(matrix -> values);
}

void printMatrix(Matrix* matrix)
{
    int row;
    int col;
    for (row = 0; row < matrix -> rows; row++)
    {
        for (col = 0; col < matrix -> columns; col++)
        {
            printf("%d\t", matrix -> values[row][col]);
        }
        printf("\n");
    }
}

void fillRandomMatrix(DATA_TYPE** matrix, int rows, int columns, int minimum, int maximum, float zeroProbability)
{
    int row;
    int col;
    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < columns; col++)
        {
            float decision = (float)rand() / (float)__INT_MAX__;
            if (decision >= zeroProbability)
            {
                matrix[row][col] = minimum + (rand() % (maximum + 1));
            }
            else
            {
                matrix[row][col] = 0;
            }
        }
    }
}

DATA_TYPE** generateRandomMatrix(int rows, int columns, int minimum, int maximum, float zeroProbability)
{
    DATA_TYPE** matrix = allocateMatrix(rows, columns);
    fillRandomMatrix(matrix, rows, columns, minimum, maximum, zeroProbability);
    return matrix;
}

Matrix* createRandomMatrix(int rows, int columns, int minimum, int maximum, float zeroProbability)
{
    DATA_TYPE** randomValues = generateRandomMatrix(rows, columns, minimum, maximum, zeroProbability);
    Matrix* randomMatrix = createMatrix(randomValues, rows, columns);
    return randomMatrix;
}

Matrix* standardSubtract(Matrix* matrix1, Matrix* matrix2)
{
    if (matrix1 -> rows != matrix2 -> rows || matrix1 -> columns != matrix2 -> columns)
    {
        return NULL;
    }

    DATA_TYPE** result = allocateMatrix(matrix1 -> rows, matrix1 -> columns);
    int row;
    int col;
    for (row = 0; row < matrix1 -> rows; row++)
    {
        for (col = 0; col < matrix1 -> columns; col++)
        {
            result[row][col] = matrix1 -> values[row][col] - matrix2 -> values[row][col];
        }
    }

    Matrix* resultMatrix = createMatrix(result, matrix1 -> rows, matrix1 -> columns);

    return resultMatrix;
}

Matrix* standardDotProduct(Matrix* matrix1, Matrix* matrix2)
{
    
    if (matrix1 -> columns != matrix2 -> rows || matrix1 == NULL || matrix2 == NULL)
    {
        return NULL;
    }

    DATA_TYPE** result = allocateMatrix(matrix1 -> rows, matrix2 -> columns);

    int matrix1Rows = matrix1 -> rows;
    int matrix1Columns = matrix1 -> columns;
    int matrix2Columns = matrix2 -> columns;
    int temp;

    int rowM1;
    int colM2;
    for (rowM1 = 0; rowM1 < matrix1 -> rows; rowM1++)
    {
        for (colM2 = 0; colM2 < matrix2 -> columns; colM2++)
        {
            int temp = 0;
            int it;
            for (it = 0; it < matrix1Columns; it++)
            {
                temp = temp + (matrix1 -> values[rowM1][it] * matrix2 -> values[it][colM2]);
            }
            result[rowM1][colM2] = temp;
        }
    }

    Matrix* resultMatrix = createMatrix(result, matrix1 -> rows, matrix2 -> columns);
    return resultMatrix;
}

Matrix* parallelDotProduct(Matrix* matrix1, Matrix* matrix2)
{
    if (matrix1 -> columns != matrix2 -> rows || matrix1 == NULL || matrix2 == NULL)
    {
        return NULL;
    }

    DATA_TYPE** result = allocateMatrix(matrix1 -> rows, matrix2 -> columns);

    int matrix1Rows = matrix1 -> rows;
    int rowM1;
    for (rowM1 = 0; rowM1 < matrix1Rows; rowM1++)
    {
        int colM2;
        int matrix1Columns = matrix1 -> columns;
        int matrix2Columns = matrix2 -> columns;
        #pragma omp parallel for private(colM2)
        for (colM2 = 0; colM2 < matrix2Columns; colM2++)
        {
            int temp = 0;
            int it;
            for (it = 0; it < matrix1Columns; it++)
            {
                temp = temp + (matrix1 -> values[rowM1][it] * matrix2 -> values[it][colM2]);
            }
            result[rowM1][colM2] = temp;
        }
    }

    Matrix* resultMatrix = createMatrix(result, matrix1 -> rows, matrix2 -> columns);
    return resultMatrix;
}

int isEqual(Matrix* matrix1, Matrix* matrix2)
{
    if (matrix1 -> columns != matrix2 -> columns || matrix1 -> rows != matrix2 -> rows)
    {
        return 0;
    }

    int row;
    int col;
    for (row = 0; row < matrix1 -> rows; row++)
    {
        for (col = 0; col < matrix1 -> columns; col++)
        {
            if (matrix1 -> values[row][col] != matrix2 -> values[row][col])
            {
                return 0;
            }
        }
    }

    return 1;
}
