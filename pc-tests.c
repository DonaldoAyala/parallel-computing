#include <stdio.h>
#include <omp.h>
#include "matrix-operations.h"

int main()
{
    srand(time(NULL));
    const int numberOfThreads = 8;
    const int rowsMatrix1 = 8;
    const int columnsMatrix1 = 8;
    const int rowsMatrix2 = 8;
    const int columnsMatrix2 = 8;
    const int minimumRandom = 1;
    const int maximumRandom = 1;
    const float zeroProbability = 0.0f;

    omp_set_num_threads(numberOfThreads);
    
    Matrix* matrix1_i = createRandomMatrix(rowsMatrix1, columnsMatrix1, minimumRandom, maximumRandom, zeroProbability);
    Matrix* matrix2_i = createRandomMatrix(rowsMatrix2, columnsMatrix2, minimumRandom, maximumRandom, zeroProbability);

    printMatrix(matrix1_i);
    printf("\n");
    printMatrix(matrix2_i);
    printf("\n");
    
    Matrix* standardMethodResult = standardDotProduct(matrix1_i, matrix2_i);
    Matrix* parallelMethodResult = parallelDotProduct(matrix1_i, matrix2_i);

    printMatrix(standardMethodResult);
    printf("\n");
    //printMatrix(parallelMethodResult);
    /*
    if (isEqual(standardMethodResult, parallelMethodResult))
    {
        printf("Results are the same\n");
    }
    else
    {
        printf("Different results\n");
    }
	*/
    
    return 0;
}