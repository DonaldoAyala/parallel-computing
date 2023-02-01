test: matrix-operations.c pc-tests.c
	gcc -fopenmp -o test.out pc-tests.c matrix-operations.c

comparison-test: parallel-vs-sequential-test.c
	gcc -fopenmp -o test.out parallel-vs-sequential-test.c

run:
	./test.out