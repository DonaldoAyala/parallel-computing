import sys
import matplotlib.pyplot as plt
import numpy as np

x = list()
y = list()

fileName = "coordinates.csv"
with open(fileName) as file:
    lines = file.readlines()
    n = int(lines[0])
    for i in range(1, n + 1):
        values = lines[i].split(',')
        x.append(float(values[0]))
        y.append(float(values[1]))
    values = lines[n + 1].split(',')
    Ax, Ay = float(values[0]), float(values[1])
    values = lines[n + 2].split(',')
    Bx, By = float(values[0]), float(values[1])

xCoordinates = np.array(x)
yCoordinates = np.array(y)

nearestPointsX = np.array([Ax, Bx])
nearestPointsY = np.array([Ay, By])

plt.plot(nearestPointsX, nearestPointsY, c='red')
plt.scatter(xCoordinates, yCoordinates, c='blue')
#plt.xticks(np.arange(0, 21, step=1))
#plt.xlabel('Cores')
#plt.ylabel('Time (ns)')
#plt.legend()
plt.show()