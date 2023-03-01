compile: nearest-points.cu
	nvcc nearest-points.cu -o exe.out

run: exe.out
	./exe.out

time-graph: nearest-points.cu time-plotter.py
	make compile && ./exe.out > times.csv && python3 time-plotter.py

coordinate-graph: nearest-points.cu coordinate-plotter.py
	make compile && ./exe.out > coordinates.csv && python3 coordinate-plotter.py