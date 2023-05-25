CC = gcc
BIN = bin/
SRC = src/
EXEC = k_means
CFLAGS = -O2 -ftree-vectorize -mavx

.DEFAULT_GLOBAL = k_means

k_means: $(SRC)k_means.c
	$(CC) $(CFLAGS) $(SRC)k_means.c -o $(BIN)$(EXEC)
	
assembly: 
	$(CC) $(CFLAGS) -S $(SRC)k_means.c -o $(BIN)$(EXEC).s
	
clean:
	rm -r $(BIN)*

run: 
	./$(BIN)$(EXEC)
	