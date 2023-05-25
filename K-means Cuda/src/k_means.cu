#include <stdio.h>
#include <stdlib.h>
#include "../includes/utils.cuh"

/// @brief Calculates euclidean distances
/// @param x1 x coordinate of the point
/// @param y1 y coordinate of the point
/// @param x2 x coordinate of the cluster
/// @param y2 y coordinate of the cluster
/// @return 
__device__ float distance(float x1, float y1, float x2, float y2){
    return (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1);
}

/// @brief Atributes each point to the nearest cluster, updating it's size, calculating the sums of it's centroid and cheeking if a point has changed cluster
/// @param data Dataset of points and labels
/// @param cluster Cluster center and it's sizes
/// @param old_center Previous cluster centers
/// @param last_label Bit mask of transitions
/// @param n number of points
/// @param k number of clusters
/// @return 
__global__ void cycle_kernel(Dataset data, Cluster cluster, Coord *old_center, int *last_label,int n, int k) {

    const int id = blockDim.x * blockIdx.x + threadIdx.x;


    const int lid = threadIdx.x;

    extern __shared__ int mem_block[];
    Coord *local_center = (Coord *) mem_block;
    int *local_label =  &(mem_block[2*k]);

    if (lid < k){
        local_center[lid] = old_center[lid];
    }

    if (id >= n)
        return;
        
    local_label[lid] = data.label[id];
    __syncthreads();

    int ind = 0;
    float min = (float) ~(1 << 31);

    float x = data.point[id].x;
    float y = data.point[id].y;

    for (int i = 0; i < k; i++) {
        float dist = distance(x, y, local_center[i].x, local_center[i].y);

        if (dist < min){
            min = dist;
            ind = i;
        }
    }

    local_label[lid] = local_label[lid] != ind ? 1 : 0;
    data.label[id] = ind;

    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if (lid < i)
            local_label[lid] += local_label[lid + i];
    }

    if (lid == 0)
        last_label[blockIdx.x] = local_label[0];
 
    atomicAdd(&(cluster.size[ind]), 1);
    atomicAdd(&(cluster.center[ind].x), x);
    atomicAdd(&(cluster.center[ind].y), y);
}

/// @brief Calculates the mean of each centroid
/// @param cluster Cluster center and it's sizes
/// @param k Number of clusters
/// @return 
__global__ void div_cluster(Cluster cluster, int k){
    int id = threadIdx.x;

    if (id >= k)
        return;

    extern __shared__ int mem_block[];
    Coord *local_cluster = (Coord *) mem_block;
    int *local_size = &(mem_block[2*k]); 
    
    local_cluster[id] = cluster.center[id];
    local_size[id] = cluster.size[id];

    __syncthreads();

    cluster.center[id].x = local_cluster[id].x / (local_size[id] == 0 ? 1 : local_size[id]);
    cluster.center[id].y = local_cluster[id].y / (local_size[id] == 0 ? 1 : local_size[id]);
}

/// @brief Sums all the elements of the bit mask, storing it in the 1st position
/// @param labels Bit mask of transitions
/// @param n length of list labels
/// @return 
__global__ void verify(int *labels, int n){
    const int id = blockDim.x * blockIdx.x + threadIdx.x; 
    
    const int lid = threadIdx.x; 

    extern __shared__ int mem_block[];
    mem_block[lid] = labels[id];

    for(int i = blockDim.x >> 1; i > 0; i >>= 1){
        if (lid < i)
            mem_block[lid] += mem_block[lid + i];
    }

    if (id == 0)
        labels[0] = 0;
    
    __syncthreads();
    if (lid == 0)
        atomicAdd(labels, mem_block[0]);

}

/// @brief k-means algorithm
/// @param argc number of arguments given
/// @param argv 1-> number of points; 2-> number of clusters
/// @return 
int main(int argc, char **argv){

    if (argc < 3){
        printf("Insuficient arguments provided");
        return 0;
    }
    
    int n = atoi(argv[1]), k = atoi(argv[2]);


    int num_threads = min(1024,32 * ((max(upper_power_2(k),64) + 31) / 32));
    int num_blocks = ((n + num_threads - 1) / num_threads);
    
    Dataset object;
    Cluster cluster;

    {
        Dataset temp = init_dataset(n);
        object = cuda_init_dataset(n, temp.point);
        cluster = cuda_init_cluster(k, object.point);
        free(temp.point);
        free(temp.label);
    }

    int *last_label = 0;
    cudaMalloc(&last_label, sizeof(int) * num_threads * ((num_blocks + num_threads - 1) / num_threads));
    CUDA_SET(int, last_label, 0, num_threads * ((num_blocks + num_threads - 1) / num_threads));

    Coord *old_center = 0;
    cudaMalloc((void **) &old_center, sizeof(struct coord) * k);
    cudaMemcpy((void *) old_center, (void *) cluster.center, sizeof(struct coord) * k, cudaMemcpyDeviceToDevice);

    int cond = 1;
    int i = -1;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    do{
        CUDA_SET(float, cluster.center, 0, 2 * k);
        CUDA_SET(int, cluster.size, 0, k);

        cycle_kernel
            <<<num_blocks, num_threads, 2 * k * sizeof(float) + num_threads * sizeof(int)>>>
            (object, cluster, old_center, last_label, n, k);

        div_cluster<<<1, num_threads, 3 * k * sizeof(float)>>>(cluster, k);
        
        cudaMemcpy((void *) old_center, (void *) cluster.center, sizeof(struct coord) * k, cudaMemcpyDeviceToDevice);

        verify<<<(num_blocks + num_threads - 1) / num_threads, num_threads, sizeof(int) * num_threads>>>(last_label, num_blocks); 

        cudaMemcpy((void *) &cond, (void *) &last_label[0], sizeof(int), cudaMemcpyDeviceToHost);
        i++;

    }while ( cond && i < 20);

    Cluster end = alloc_cluster(k);
    cudaMemcpy((void *) end.center, (void*) cluster.center, k * sizeof(struct coord), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *) end.size, (void*) cluster.size, k * sizeof(int), cudaMemcpyDeviceToHost);


    printf("N = %d, K = %d\n",n,k);
    for(int j = 0; j < k; j++)
        printf("Center: (%.3f,%.3f) : Size: %d\n",end.center[j].x,end.center[j].y,end.size[j]);
    printf("Iterations: %d\n",i);

    free(end.center);
    free(end.size);
    cudaFree(object.point);
    cudaFree(object.label);
    cudaFree(cluster.center);
    cudaFree(cluster.size);
    cudaFree(last_label);
    cudaFree(old_center);

    return 0;
}