CC = nvcc
BIN = bin/
SRC = src/
INCLUDES = includes/
EXEC = k_means
CFLAGS = -O3 -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets

.DEFAULT_GLOBAL = k_means

k_means: $(SRC)k_means.cu $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.cu $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.cu $(INCLUDES)utils.cuh
	$(CC) $(CFLAGS) -dc $(SRC)utils.cu -odir $(BIN)
		
clean:
	rm -r $(BIN)*

run: 
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)