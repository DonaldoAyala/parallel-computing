#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <stdint.h>
#include "matrix-operations.h"

void initRandomSeed()
{
    srand(time(NULL));
}

// Executes the function at the beginning of the code
void __attribute__((constructor)) initRandomSeed(); 

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

void test(int numberOfThreads, int matrixSize)
{
    // Test parameters
    const int rowsMatrix1 = matrixSize;
    const int columnsMatrix1 = matrixSize;
    const int rowsMatrix2 = matrixSize;
    const int columnsMatrix2 = matrixSize;
    const int minimumRandom = 1;
    const int maximumRandom = 1;
    const float zeroProbability = 0.0f;
    
    // Setting number of threads to use
    omp_set_num_threads(numberOfThreads);
    printf("%d,", omp_thread_count());
    
    // Matrix creation
    Matrix* matrix1_i = createRandomMatrix(rowsMatrix1, columnsMatrix1, minimumRandom, maximumRandom, zeroProbability);
    Matrix* matrix2_i = createRandomMatrix(rowsMatrix2, columnsMatrix2, minimumRandom, maximumRandom, zeroProbability);
    
    // To measure execution times
    struct timespec start;
    struct timespec end;
    uint64_t standardCalculationTime = 0;
    uint64_t parallelCalculationTime = 0;

    // Measures time of parallel matrix dot product
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    Matrix* parallelMethodResult = parallelDotProduct(matrix1_i, matrix2_i);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    parallelCalculationTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("%ld,", parallelCalculationTime);
    
    // Measures time of sequential matrix dot product
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    Matrix* standardMethodResult = standardDotProduct(matrix1_i, matrix2_i);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    standardCalculationTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("%ld,", standardCalculationTime);

    // Print results
    if (isEqual(parallelMethodResult, standardMethodResult) == 1)
    {
        printf("equals\n");
    }
    else
    {
        printf("not equals\n");
    }
}

int main(int argsc, char** args)
{
    if (argsc < 2)
    {
        printf("Provide size of square matrix\n");
        return 1;
    }

    // Variating the matrix size
    int matrixSize = atoi(args[1]);
    
    int cores;
    for (cores = 1; cores <= 20 ; cores++)
    {
        test(cores, matrixSize);
    }
    

    return 0;
}