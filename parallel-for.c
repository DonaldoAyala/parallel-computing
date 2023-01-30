#include <stdio.h>
#include <time.h>
#include <omp.h>

int main()
{
    const int size = 100;
    int mat[size][size];

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        int j;
        for (j = 0; j < 10; j++)
        {
            int threadNumber =  omp_get_num_threads(numberOfThreads);
            printf("");
        }
    }

    return 0;
}