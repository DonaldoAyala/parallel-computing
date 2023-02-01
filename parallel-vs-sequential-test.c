#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <stdint.h>

int main (int argc, char* argsv[])
{
    const int size = 1000;
    int* matrix = (int*) malloc(size * sizeof(int));

    omp_set_num_threads(8);

    int i;
    //clock_t start;
    //clock_t end;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //start = clock();
    #pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        matrix[i] = 0;
        int j;
        for (j = 0; j < size * size; j++)
        {
            matrix[i] += 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    //end = clock();
    printf("Dynamic memory\n");
    printf("P execution time: %ld\n", delta_us);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //start = clock();
    for (i = 0; i < size; i++)
    {
        matrix[i] = 0;
        int j;
        for (j = 0; j < size * size; j++)
        {
            matrix[i] += 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    //end = clock();

    printf("S execution time: %ld\n", delta_us);

    for (i = 0; i < 10; i++)
    {
        printf("%d  ", matrix[i]);
    }
    printf("\n");

    return 0;
}