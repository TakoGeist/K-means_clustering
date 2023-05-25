/* k_means.c
Project coded by: 
	Diogo Ramos, a95109
	Gabriel Costa, a94893
	
	UMinho
	11/2022
*/


#include <stdio.h>
#include <stdlib.h>

#define N 10000000
#define K 4

typedef struct Coord{
    float x;
    float y;
} Coord;

/* Initializes a dataset of coordinates with random values between 0 and 1. 
   Initializes the cluster with values from the dataset.*/
void init(Coord * restrict object, Coord * restrict cluster) {
    
    srand(10);
	
    for(int i = 0; i < N; i++) {
        object[i].x = (float) rand() / (float) RAND_MAX;
        object[i].y = (float) rand() / (float) RAND_MAX;
    }

    for(int i = 0; i < K; i++) {
        cluster[i] = object[i];
    }
}

/* Distribution function to assign each point to it's closest cluster*/
void cycle(Coord * restrict object, int * restrict label, Coord * restrict cluster, int * restrict size){
    int i,j;

    for(i = 0; i < K; i++){
        size[i] = 0;
    }

    float* dist = malloc(sizeof(float) * 2 * K);
    
    for(i = 0; i < N-1; i+=2){
        float x0 = object[i].x;
        float y0 = object[i].y;
        float x1 = object[i + 1].x;
        float y1 = object[i + 1].y;

        /* Vectorized sub-routine to calculate 2*K distances to dist array*/
        for(j = 0; j < K; j++){
            dist[j] = (cluster[j].x - x0)*(cluster[j].x - x0) + (cluster[j].y - y0)*(cluster[j].y - y0);
            dist[K + j] = (cluster[j].x - x1)*(cluster[j].x - x1) + (cluster[j].y - y1)*(cluster[j].y - y1);
        }
		
        int ind0 = 0;
        int ind1 = 0;
        float min0 = dist[0];
        float min1 = dist[K];

        /* Sub-routine to calculate smallest distance relative to each point*/
        for(j = 1; j < K; j++){
            if (dist[j] < min0){
                min0 = dist[j];
                ind0 = j;
            }

            if (dist[K + j] < min1){
                min1 = dist[K + j];
                ind1 = j;
            }
        }
        size[ind0] += 1;    
        size[ind1] += 1;
        label[i] = ind0;    
        label[i+1] = ind1;    
    }

    /*Sub-routine to guarantee calculation of every point (necessary because of unroll,
     should remain equal to for loop above)*/
    for(i; i < N; i++){        
        float x0 = object[i].x;
        float y0 = object[i].y;
        
		for(j = 0; j < K; j++){
            dist[j] = (cluster[j].x - x0)*(cluster[j].x - x0) + (cluster[j].y - y0)*(cluster[j].y - y0);
        }
		
        int ind0 = 0;
        float min0 = dist[0];
        for(j = 1; j < K; j++){
            if (dist[j] < min0){
                min0 = dist[j];
                ind0 = j;
            }

        }
        size[ind0] += 1;    
        label[i] = ind0;    
    }
    free(dist);
}

/* Recalculates the center of each cluster based on the men of all it's points*/
int recalc(Coord * restrict object, int * restrict label, Coord * restrict cluster, int * restrict size) {
    int i,j,cond = 0;
    Coord centroid[K] = {{0,0}};

    for (i = 0; i<N; i++) {
        int ind = label[i];
        centroid[ind].x += object[i].x;
        centroid[ind].y += object[i].y;
    }

    for(j = 0; j < K; j++){
        centroid[j].x = centroid[j].x/size[j];
        centroid[j].y = centroid[j].y/size[j];
    }

    /*Verification of cluster invariance to determine whether the clustering has concluded*/
    for (j = 0; j<K && (cluster[j].x == centroid[j].x || cluster[j].y == centroid[j].y); j++);
    if (j != K) cond++;

    for (j = 0; j<K; j++) {
        cluster[j] = centroid[j];
    }

    return cond;
}

int main(){
    Coord *object = malloc(sizeof(struct Coord) * N);
    int *label = malloc(sizeof(int) * N);

    Coord *cluster = malloc(sizeof(struct Coord) * K);
    int *size = malloc(sizeof(int) * K);

    init(object, cluster);

    int cond;
    int i = -1;

    do{
        cycle(object, label, cluster, size);
        cond = recalc(object, label, cluster, size);
        i++;
    }while (cond);

    printf("N = %d, K = %d\n",N,K);
    for(int j = 0; j < K; j++)
    printf("Center: (%.3f,%.3f) : Size: %d\n",cluster[j].x,cluster[j].y,size[j]);
    printf("Iterations: %d\n",i);

    free(object);
    free(cluster);
    free(size);
    free(label);

    return 0;
}

