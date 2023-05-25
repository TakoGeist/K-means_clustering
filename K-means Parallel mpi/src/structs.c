#include <stdlib.h>

#include "structs.h"

Dataset init_dataset(int size){

    Dataset new = {malloc(sizeof(struct coord) * size),
                   malloc(sizeof(int) * size)};

    srand(10);

    for(int i = 0; i < size; i++){
        new.point[i].x = (float) rand() / RAND_MAX;
        new.point[i].y = (float) rand() / RAND_MAX;
    }

    return new;
}

Cluster init_cluster(int number, Coord *point){

    Cluster new = {malloc(sizeof(struct cluster) * number),
                   malloc(sizeof(int) * number)};

    for(int i = 0; i < number; i++){
        new.center[i] = point[i];
    }

    return new;
}

