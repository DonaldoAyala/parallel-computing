test: matrix-operations.c pc-tests.c matrix-operations.h
	gcc -fopenmp -o test.out pc-tests.c matrix-operations.c

comparison-test: parallel-vs-sequential-test.c
	gcc -fopenmp -o test.out parallel-vs-sequential-test.c

run:
	./test.out

graph: plotter.py
	make && ./test.out $(size) > size$(size).csv && python3 plotter.py $(size)
