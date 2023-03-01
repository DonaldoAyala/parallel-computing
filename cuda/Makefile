compile: nearest-points.cu
	nvcc nearest-points.cu -o exe.out

run: exe.out
	./exe.out

graph: nearest-points.cu plotter.py
	make compile && ./exe.out > coordinates.csv && python3 plotter.py