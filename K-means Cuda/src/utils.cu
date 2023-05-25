#include <stdlib.h>
#include "../includes/utils.cuh"

Dataset init_dataset(int size){
    
    Dataset data = {(Coord *) malloc(sizeof(struct coord) * size),
                    (int *) malloc(sizeof(int) * size)};

    srand(10);

    for(int i = 0; i < size; i++){
        data.point[i].x = (float) rand() / RAND_MAX;
        data.point[i].y = (float) rand() / RAND_MAX;
    }

    return data;
}

Cluster init_cluster(int size, Coord *point){

    Cluster cluster = {(Coord *)malloc(sizeof(struct cluster) * size),
                   (int *)malloc(sizeof(int) * size)};

    for(int i = 0; i < size; i++){
        cluster.center[i] = point[i];
    }

    return cluster;
}

Cluster alloc_cluster(int size){
    return {(Coord *)malloc(sizeof(struct cluster) * size),
            (int *)malloc(sizeof(int) * size)};
}

Dataset cuda_init_dataset(int size, Coord *data){

    Coord *point = 0;
    int *label = 0;

    cudaMalloc(&point, sizeof(struct coord) * size);
    cudaMalloc(&label, sizeof(int) * size);
    cudaMemcpy((void *) point, (void *) data, sizeof(struct coord) * size, cudaMemcpyHostToDevice);
    CUDA_SET(int, label, -1, size);

    return {point, label};
}

Cluster cuda_init_cluster(int size, Coord *data){
    Coord *center = 0;
    int *num = 0;

    cudaMalloc(&center, sizeof(struct coord) * size);
    cudaMalloc(&num, sizeof(int) * size);
    cudaMemcpy((void *) center, (void *) data, sizeof(struct coord) * size, cudaMemcpyDeviceToDevice);
    CUDA_SET(int, num, 0, size);

    return {center, num};

}

unsigned int upper_power_2(unsigned int v){
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}