import sys
import matplotlib.pyplot as plt
import numpy as np

pointsNumber = list()
parallelExecutionTime = list()
sequentialExecutionTime = list()


fileName = "times.csv"
with open(fileName) as file:
    lines = file.readlines()
    for line in lines:
        values = line.split(',')
        pointsNumber.append(float(values[0]))
        parallelExecutionTime.append(float(values[1]))
        sequentialExecutionTime.append(float(values[2]))

xCoordinates = np.array(pointsNumber)
yCoordinatesParallel = np.array(parallelExecutionTime)
yCoordinatesSequential = np.array(sequentialExecutionTime)

plt.scatter(xCoordinates, yCoordinatesParallel, c='red', label='Parallel execution')
plt.scatter(xCoordinates, yCoordinatesSequential, c='blue', label='Sequential execution')
#plt.xticks(np.arange(0, 21, step=1))
plt.xlabel('Number of points')
plt.ylabel('Time (ns)')
plt.grid()
plt.legend()
plt.show()