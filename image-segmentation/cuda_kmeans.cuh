#ifndef _CUDA_KMEANS_
#define _CUDA_KMEANS_
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

static void HandleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HCUDAERR( err ) (HandleCudaError( err, __FILE__, __LINE__ ))

/* Setup cura state */
__global__
void setupKernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

/* Sets the centroid to a random point */
__device__
void randomPoint(curandState *state, const int clusID, const int dims,
float *clusters, const int nelems, const float *elems) {
    int cPos = clusID*dims;
    // Generate random
    // https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
    // uniform dist
    float frand;
    int ePos;
    for(int c = 0; c <= clusID; c++)
        // Do this for clusID iterations
        frand = curand_uniform(&(state[clusID]));
        frand *= nelems-1;
        ePos = (int)truncf(frand);
        ePos *= dims;
    for (int i = 0; i < dims; i++)
        clusters[cPos+i] = elems[ePos+i];
}

/* Function to group the elements in its nearest cluster.
 * Returns the distance to said cluster. */
__device__
float grouping(const int elemID, const int dims, const int nclusters,
const float *clusters, const float *elems, int *elemClus) {
    float dist, elemDist = 99999;
    int cPos, ePos = elemID*dims;

    // Start by considering as part of the first cluster
    for (int i = 0; i < nclusters; i++) {
        // Calculate the position of the cluster in the linearized matrix
        cPos = i*dims;

        // Calculate distance between vectors
        dist = 0;
        for (int j = 0; j < dims; j++)
            dist += (clusters[cPos+j]-elems[ePos+j])*
                (clusters[cPos+j]-elems[ePos+j]);
        dist = sqrtf(dist);

        // Calculate total distance
        //dist = normf(dims, ddist);

        // See if it is less than the distance to the current cluster
        if (dist < elemDist) {
            elemClus[elemID] = i;
            elemDist = dist;
        }
    }

    return elemDist;
}

/* Function to update the centroid to be in the middle of the calculated
 * clusters.
 * returns: the distance moved */
__device__
float updateCentroid(const int clusID, const int dims, float *clusters,
const int nelems, const float *elems, int *elemClus) {
    float avg, dis = 0;
    int cPos = clusID*dims;
    int counter = 0;

    for (int i = 0; i < dims; i++) {
        avg = 0;
        for (int j = 0; j < nelems; j++)
            if (elemClus[j] == clusID) {
                avg += elems[dims*j + i];
                counter++;
            }
        avg /= counter;
        dis += fabsf(clusters[cPos+i] - avg);
        clusters[cPos + i] = avg;
    }

    return dis;
}

__device__
float calcMean(const int n, const float *elems) {
    float mean = 0;
    for (int i = 0; i < n; i++) mean += elems[i]/n;
    return mean;
}

/* Function to check if the clusters moved.
 * Returns a boolean (false -> if they moved, true -> they didnt) */
__device__
int testChange(const int nclusters, const float *distances){
    for (int i = 0; i < nclusters; i++) if(distances[i] > 0) return false;
    return true;
}

/* Runs the kmean algorithm:
 * Inputs:
 *   - dims <int>: Number of dimensions
 *   - epochs <int>: Number of iterations
 *   - limit <int>: Number of updates to centroid until it gives up on finding
                the sweet spot.
 *   - nclusters <int>: Number of clusters to calculate
 *   - clusters <float*>: Pointer where all the clusters will be saved.
 *              size(float[nclusters*dims])
 *   - nelems <int>: Number of elems received
 *   - elems <float*>: Pointer to all the data. size(float[nelems*dims])
 *   - elemClus <int*>: Array where the relation elem-cluster is saved.
 *              size(int[nelems])
 *   - entropy <float*>: Pointer to where to save the calculated entropy per
 *              iteration. size(float[epochs])
 */
__global__
void kmeans(curandState *state, const int dims, const int epochs, const int limit,
const int nclusters, float *clusters, const int nelems, float *elems,
int *elemClus, float *entropy, float *elemDis, float *movedDis, float *bC) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    for (int i = 0; i < epochs; i++) {
        if (tid < nclusters)
            randomPoint(state+tid, tid, dims, clusters, nelems, elems);
        for (int j = 0; j < limit; j++) {
            // Calculate 
            if (tid < nelems)
                elemDis[tid] = grouping(tid, dims, nclusters, clusters, elems,
                    elemClus);

            __syncthreads();

            if (tid < nclusters)
                movedDis[tid] = updateCentroid(tid, dims, clusters, nelems,
                    elems, elemClus);

            __syncthreads();

            if (tid < 1) {
                // Calculate entropy
                entropy[epochs] = calcMean(nelems, elemDis);
                if (testChange(nclusters, movedDis)) break;
            }
        }
        // TODO: copy to bestCluster
    }
}
#endif
