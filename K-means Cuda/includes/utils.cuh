#define CUDA_SET(type,pointer,value,size) (\
    cudaMemset(pointer, (type) value, size * sizeof(type)))

typedef struct coord{
    float x;
    float y;
} Coord;

typedef struct dataset{
    Coord *point;
    int *label;
} Dataset;

typedef struct cluster{
    Coord *center;
    int *size;
} Cluster; 

Dataset init_dataset(int);

Cluster init_cluster(int, Coord *);

Dataset cuda_init_dataset(int, Coord *);

Cluster cuda_init_cluster(int, Coord *);

Cluster alloc_cluster(int);

unsigned int upper_power_2(unsigned int );