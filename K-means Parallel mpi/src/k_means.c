
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "structs.h"

/* Distribution function to assign each point to it's closest cluster*/
void cycle(Dataset * object, Cluster * cluster, int n, int k, int threads){
    int cond=0;

    #pragma omp parallel num_threads(threads)
    {

    int j;
    int *thread_size = (int *) malloc(sizeof(int) * k);
    float *dist = (float *) malloc(sizeof(float) * k);

    zero(int, thread_size, k);

    #pragma omp for 
    for(int i = 0; i < n; i++){

        point_distance(object->point[i], cluster->center, dist, k);

        int ind0 = 0;
        float min0 = dist[0];

        for(j = 1; j < k; j++){
            if (dist[j] < min0){
                min0 = dist[j];
                ind0 = j;
            }
        }

        thread_size[ind0] += 1;    
        object->label[i] = ind0;    
    }
    
    #pragma omp critical
    {
        for(int i = 0; i < k; i++){
            cluster->size[i] += thread_size[i];
        }
    }

    free(thread_size);
    free(dist);
    }
}

/* Recalculates the center of each cluster based on the mean of all it's points*/
int centroid_calculation(Dataset * object, Cluster * cluster, int n, int k, int threads) {
    int i, j, cond = 0;

    Coord *centroid = (Coord *) malloc(sizeof(struct coord) * k);

    zero(float, centroid, 2*k);

    float *aux_centroid = (float *) centroid;
    float *aux_point = (float *) object->point;

    #pragma omp parallel num_threads(threads)
    {
        float *thread_centroid = (float *) malloc(sizeof(float) * 2 * k);
        int ti;

        zero(float, thread_centroid, 2*k);
        
        #pragma omp for
        for (ti = 0; ti < n; ti++) {
            int ind = object->label[ti];
            thread_centroid[ind*2]     += aux_point[ti*2];
            thread_centroid[ind*2 + 1] += aux_point[ti*2 + 1];
        }

        #pragma omp critical
        {
        for(ti = 0; ti < 2 * k; ti++){
            aux_centroid[ti] += thread_centroid[ti];
        }
        }

        free(thread_centroid);
    }

    for(j = 0; j < k; j++){
        centroid[j].x = centroid[j].x/cluster->size[j];
        centroid[j].y = centroid[j].y/cluster->size[j];
    }

    for (j = 0; j<k; j++)
    if ((cluster->center[j].x != centroid[j].x || cluster->center[j].y != centroid[j].y)) cond++;


    for (j = 0; j<k; j++) {
        cluster->center[j] = centroid[j];
    }

    free(centroid);

    return cond;
}


int main(int argc, char **argv){

    if (argc < 3){
        printf("Insuficient arguments provided");
        return 0;
    }

    int n = atoi(argv[1]), k = atoi(argv[2]);

    int num_threads = 1;

    if (argc > 3){
        num_threads = atoi(argv[3]);
    }

    Dataset object = init_dataset(n);
    Cluster cluster = init_cluster(k, object.point);

    int cond;
    int i = -1;

    do{
        zero(int, cluster.size, k);
        cycle(&object, &cluster, n, k, num_threads);
        cond = centroid_calculation(&object, &cluster, n, k, num_threads);
        i++;
    }while (cond && i < 20);

    printf("N = %d, K = %d\n",n,k);
    for(int j = 0; j < k; j++)
        printf("Center: (%.3f,%.3f) : Size: %d\n",cluster.center[j].x,cluster.center[j].y,cluster.size[j]);
    printf("Iterations: %d\n",i);

    return 0;
}
