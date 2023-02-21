import sys
import matplotlib.pyplot as plt
import numpy as np

plot = 1
size = sys.argv[1]
parallel = list()
sequential = list()
cores = list()

fileName = "size" + str(size) + ".csv"
with open(fileName) as file:
    lines = file.readlines()
    for line in lines:
        values = line.split(',')
        cores.append(int(values[0]))
        parallel.append(int(values[1]))
        sequential.append(int(values[2]))

parallelPoints = np.array(parallel)
sequentialPoints = np.array(sequential)
coresPoints = np.array(cores)

print(coresPoints)
print(parallelPoints)

plt.title('Square matrix of size ' + str(size))
plt.plot(coresPoints, parallelPoints, label="Parallel")
plt.plot(coresPoints, sequentialPoints, label='Sequential')
plt.xticks(np.arange(0, 21, step=1)) 
plt.xlabel('Cores')
plt.ylabel('Time (ns)')
plt.legend()
plt.grid()

size = size * 10
plot += 1
plt.show()
