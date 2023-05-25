CC = gcc
BIN = bin/
SRC = src/
EXEC = k_means
THREADS = 6
CFLAGS = -O3 -mavx -fopenmp -lm

.DEFAULT_GLOBAL = k_means

k_means: $(SRC)k_means.c
	$(CC) $(CFLAGS) $(SRC)k_means.c $(SRC)structs.c -o $(BIN)$(EXEC)
		
clean:
	rm -r $(BIN)*

runseq: 
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

runpar: 
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)
	