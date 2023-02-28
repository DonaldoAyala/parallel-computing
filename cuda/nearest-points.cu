#include <stdio.h>
#include <stdlib.h>
#define N 4
const int lowerLimit = -100;
const int upperLimit = 100;

struct Lock {

    int *d_state;

    // --- Constructor
    Lock(void) {
        int h_state = 0;                                        // --- Host side lock state initializer
        cudaMalloc((void **)&d_state, sizeof(int));  // --- Allocate device side lock state
        cudaMemcpy(d_state, &h_state, sizeof(int), cudaMemcpyHostToDevice); // --- Initialize device side lock state
    }

    // --- Destructor
    __host__ __device__ ~Lock(void) { 
    #if !defined(__CUDACC__)
        cudaFree(d_state); 
    #else

    #endif  
    }

    // --- Lock function
    __device__ void lock(void) { while (atomicCAS(d_state, 0, 1) != 0); }

    // --- Unlock function
    __device__ void unlock(void) { atomicExch(d_state, 0); }
};


struct point
{
    double x;
    double y;
    
    point(double x, double y) : x(x), y(y) {}
};

__global__ void findNearestPointWithLocks(point* points, point* nearestPoints, double* nearestPointsDistances, int* blockLocks)
{
    if (blockIdx.x == 0)
    {
        int id = blockIdx.x;
        printf("Lock at %d with value %d\n", threadIdx.x, blockLocks[id]);
        while (atomicCAS(&blockLocks[id], 0, 1) != 0) ; // lock();
        printf("Thread %d aquired the lock\n", threadIdx.x);
        nearestPointsDistances[id] += 1;
        atomicExch(&blockLocks[id], 0); // unlock();
    }
}

__global__ void findNearestPoint(point* points, point* nearestPoints, double* nearestPointsDistances)
{
    int blockId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int threadId = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x + blockId * (blockDim.x * blockDim.y * blockDim.z);    
    
    point A = points[threadId];
    for (int i = 0; i < N; i++) 
    {
        if (threadId == i) continue;
        point B = points[i];
        double distance = sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
        if (distance < nearestPointsDistances[threadId]) 
        {
            //printf("Closest to point (%f,%f) is (%f,%f) at euclidian distance of %f\n", A.x, A.y, B.x, B.y, distance);
            nearestPointsDistances[threadId] = distance;
            nearestPoints[threadId] = B;
        }
    }
}

int main () {
    srand(time(NULL));
    cudaDeviceReset();

    dim3 gridSize(1);
    dim3 blockSize(N);

    // Declaring and initializing host variables
    int* blockLocks = (int*) malloc(N * sizeof(int));
    point* points = (point*) malloc(N * sizeof(point));
    point* nearestPoints = (point*) malloc(N * sizeof(point));
    double* nearestPointsDistances = (double*) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        blockLocks[i] = 0;
        points[i].x = lowerLimit + rand() % ((upperLimit - lowerLimit) + 1);
        points[i].y = lowerLimit + rand() % ((upperLimit - lowerLimit) + 1);
        nearestPointsDistances[i] = (upperLimit - lowerLimit) + 10; // The maximum distance plus an extra 10
        //printf("(%f,%f)\n", points[i].x, points[i].y);
    }

    // Create device variables and allocate memory on device
    point* points_d;
    point* nearestPoints_d;
    int* blockLocks_d;
    double* nearestPointsDistances_d;
    cudaMalloc(&points_d, N * sizeof(point));
    cudaMalloc(&nearestPoints_d, N * sizeof(point));
    cudaMalloc(&blockLocks_d, N * sizeof(int));
    cudaMalloc(&nearestPointsDistances_d, N * sizeof(double));

    // Copy values to devices
    cudaMemcpy(points_d, points, N * sizeof(point), cudaMemcpyHostToDevice);
    cudaMemcpy(nearestPointsDistances_d, nearestPointsDistances, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(blockLocks_d, blockLocks, N * sizeof(int), cudaMemcpyHostToDevice);

    // Execute kernel
    findNearestPoint<<<gridSize, blockSize>>>(points_d, nearestPoints_d, nearestPointsDistances_d);
    
    // Wait till every thread has finished
    cudaDeviceSynchronize();

    // Retrieve values from device memory
    cudaMemcpy(nearestPointsDistances, nearestPointsDistances_d, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(nearestPoints, nearestPoints_d, N * sizeof(point), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++)
    {
        printf("Closest to point (%f,%f) is (%f,%f) at euclidian distance of %f\n",
        points[i].x, points[i].y, nearestPoints[i].x, nearestPoints[i].y, nearestPointsDistances[i]);
    }

    // Free all resources
    cudaDeviceReset();

    return 0;
}