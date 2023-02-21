#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void getPi()
{
    double pi;
    long steps = 1000000000;
    double step = 1.0/(double) steps;
    double start = omp_get_wtime();

    double sum = 0;
    #pragma omp parallel
    {
        int i;
        double x;
        #pragma omp for reduction(+:sum) // Creates a copy of sum for each thread, and joins the results of each chunk of iterations at the end
        for (i = 0; i < steps; i++)
        {
            x = (i + 0.5) * step;
            sum = sum + 4.0/(1.0 + x*x);
        }
    }
    // At this point, sum has already the acumulated values of the sum of each thread.
    pi = step * sum;

    double end = omp_get_wtime();

    printf("Pi value %f in %f\n", pi, end - start);
}

int main()
{
    double pi;

    int numThreads = 8;
    omp_set_num_threads(numThreads);

    long steps = 1000000000;
    double step = 1.0/(double) steps;
    int chunkSize = steps / numThreads;
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();

        double x;
        int i;
        double sum = 0;

        for (i = 0; i < chunkSize; i++)
        {
            x = (i + chunkSize*threadId + 0.5) * step;
            sum = sum + 4.0/(1.0 + x*x);
        }
        
        #pragma omp critical
        pi += step * sum;
    }
    double end = omp_get_wtime();

    printf("Pi value %f in %f\n", pi, end - start);
    getPi();
}