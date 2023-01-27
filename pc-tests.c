#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "matrix-operations.h"

#define MATRIX_SIZE 1000

void initRandomSeed()
{
    srand(time(NULL));
}

// Executes the function at the beginning of the code
void __attribute__((constructor)) initRandomSeed(); 

int main()
{
    const int numberOfThreads = 20;
    const int rowsMatrix1 = MATRIX_SIZE;
    const int columnsMatrix1 = MATRIX_SIZE;
    const int rowsMatrix2 = MATRIX_SIZE;
    const int columnsMatrix2 = MATRIX_SIZE;
    const int minimumRandom = 1;
    const int maximumRandom = 1;
    const float zeroProbability = 0.0f;

    omp_set_num_threads(numberOfThreads);
    
    Matrix* matrix1_i = createRandomMatrix(rowsMatrix1, columnsMatrix1, minimumRandom, maximumRandom, zeroProbability);
    Matrix* matrix2_i = createRandomMatrix(rowsMatrix2, columnsMatrix2, minimumRandom, maximumRandom, zeroProbability);
    
    clock_t start;
    clock_t end;
    double standardCalculationTime = 0;
    double parallelCalculationTime = 0;

    start = clock();
    Matrix* standardMethodResult = standardDotProduct(matrix1_i, matrix2_i);
    end = clock();
    standardCalculationTime = end - start;

    start = clock();
    Matrix* parallelMethodResult = parallelDotProduct(matrix1_i, matrix2_i);
    end = clock();
    parallelCalculationTime = end - start;

    printf("Standard: %f - Parallel: %f\nDifference: %f\n", 
    standardCalculationTime / (double)CLOCKS_PER_SEC, 
    parallelCalculationTime / (double)CLOCKS_PER_SEC, 
    (standardCalculationTime - parallelCalculationTime) / (double)CLOCKS_PER_SEC);
    
    //printMatrix(standardMethodResult);
    //printf("\n");
    //printMatrix(parallelMethodResult);
    if (isEqual(parallelMethodResult, standardMethodResult) == 1)
    {
        printf("Same\n");
    }
    else
    {
        printf("Not same\n");
    }

    return 0;
}