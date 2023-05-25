
/*Macro to set a array of size "size" and type "type" to all 0's */
#define zero(type, pointer, size) ({type *new = (type *) pointer;for(int i = 0; i < size; i++) new[i] = 0;})

typedef struct coord{
    float x;
    float y;
} Coord;

typedef struct dataset{
    Coord *const point;
    int *const label;
} Dataset;

typedef struct cluster{
    Coord *const center;
    int *const size;
} Cluster; 

Dataset init_dataset(int);

Cluster init_cluster(int, Coord *);

/* Inlined function to calculate the distance of a point to all it's clusters*/
inline __attribute__((always_inline)) void point_distance(Coord point, Coord *cluster, float *dist, int k){
    float *aux_center = (float *) cluster;

    for(int i = 0; i < k; i++){
        dist[i] = (aux_center[2*i] - point.x)*(aux_center[2*i] - point.x) 
                + (aux_center[2*i + 1] - point.y)*(aux_center[2*i + 1] - point.y);
    }

}