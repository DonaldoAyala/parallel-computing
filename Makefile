test: matrix-operations.c pc-tests.c
	gcc -fopenmp -o test.out pc-tests.c matrix-operations.c

run:
	./test.out