#ifndef _CUDA_KMEANS_
#define _CUDA_KMEANS_
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define TPB 512
#define MAXDISPLAY 10

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
void gpu_kmeans(const int dims, const int epoch, const int limit,
const int nclusters, float *clusters, const int nelems, float *elems,
int *elemClus, float *entropy, float *elemDis, float *movedDis) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    for (int j = 0; j < limit+1; j++) {
        // Calculate 
        if (j > 0) {
            if (tid < nclusters)
                movedDis[tid] = updateCentroid(tid, dims, clusters, nelems,
                    elems, elemClus);

            __syncthreads();
        }
        if (tid < nelems)
            elemDis[tid] = grouping(tid, dims, nclusters, clusters, elems,
                elemClus);

        __syncthreads();

        if (tid < 1) {
            // Calculate entropy
            entropy[epoch] = calcMean(nelems, elemDis);
            if (testChange(nclusters, movedDis)) break;
        }
    }
}

void logKmeansEpoch(FILE *log, int dims, int nclusters, float *clusters,
int nelems, float *data, int *elemClus, float entropy, float *elemDis) {
    fprintf(log, "Resulting clusters:\n");
    for (int i = 0; i < nclusters; i++) {
        fprintf(log, "\t#%i: ", i);
        for (int j = 0; j < dims; j++)
            fprintf(log, "%.1f ", clusters[i*dims+j]);
        fprintf(log, "\n");
    }

    fprintf(log, "Entropy: %f\n", entropy);

    fprintf(log, "Elements:\n");
    // Log elements and its distance
    for (int i = 0; i < nelems && i < MAXDISPLAY; i++) {
        fprintf(log, "\t#%i: ", i);
        for (int j = 0; j < dims; j++)
            fprintf(log, "%.1f ", data[i*dims+j]);
        fprintf(log, "Cluster (%i, %.1f)\n", elemClus[i], elemDis[i]);
    }
}

void copyToHost(int csize, float *clusters, float *gpu_clusters, int nelems,
int *elemClus, int *gpu_elemClus, float *elemDis, float *gpu_elemDis,
int epochs, float *entropy, float *gpu_entropy) {
    // Copy GPU data to HOST
    HCUDAERR(cudaMemcpy(clusters, gpu_clusters, sizeof(float)*csize,
        cudaMemcpyDeviceToHost));
    HCUDAERR(cudaMemcpy(elemClus, gpu_elemClus, sizeof(int)*nelems,
        cudaMemcpyDeviceToHost));

    HCUDAERR(cudaMemcpy(elemDis, gpu_elemDis, sizeof(float)*nelems,
        cudaMemcpyDeviceToHost));

    HCUDAERR(cudaMemcpy(entropy, gpu_entropy, sizeof(float)*epochs,
        cudaMemcpyDeviceToHost));
}


int * kmeans(const int dims, const int epochs, const int limit,
const int nclusters, const int nelems, float *data, FILE *log,
float *gpu_clusters, int *gpu_elemClus) { // Added this so i can use them later
    // HOST data
    float *clusters, *entropy, *elemDis, bestEntropy;
    int size, *elemClus, r;

    // GPU data
//    float *gpu_clusters;
    float *gpu_data, *gpu_entropy, *gpu_elemDis, *gpu_movedDis;
//    int *gpu_elemClus;

    size = nelems*dims;

    // Allocate HOST data
    clusters = (float *)malloc(sizeof(float)*nclusters*dims);
    elemDis = (float *)malloc(sizeof(float)*nelems);
    entropy = (float *)malloc(sizeof(float)*nelems);
    elemClus = (int *)malloc(sizeof(int)*nelems);

    // Allocate GPU data
    fprintf(log, "Allocating memory in GPU...\n");
    HCUDAERR(cudaMalloc((void**) &gpu_data, sizeof(float)*size));
//    HCUDAERR(cudaMalloc((void**) &gpu_clusters, sizeof(float)*nclusters*dims));
//    HCUDAERR(cudaMalloc((void**) &gpu_elemClus, sizeof(int)*nelems));
    HCUDAERR(cudaMalloc((void**) &gpu_entropy, sizeof(float)*epochs));
    HCUDAERR(cudaMalloc((void**) &gpu_elemDis, sizeof(float)*nelems));
    HCUDAERR(cudaMalloc((void**) &gpu_movedDis, sizeof(float)*nclusters));

    // Upload data to GPU
    fprintf(log, "Uploading data to GPU...\n");
    HCUDAERR(cudaMemcpy(gpu_data, data, sizeof(float)*size,
        cudaMemcpyHostToDevice));

    // Call kmeans
    fprintf(log, "Applying kmeans...\n");
    for (int i = 0; i < epochs; i++) {
        fprintf(log, "\n\nIteration (%i/%i)\n", i, epochs);
        // Random starting points
        for (int j = 0; j < nclusters; j++) {
            r = rand()%nelems;
            fprintf(log, "Start point %i: elem %i < ", j, r);
            for (int k = 0; k < dims; k++) {
                clusters[j*dims+k] = data[r*dims+k];
                fprintf(log, "%f ", clusters[j*dims+k]);
            }
            fprintf(log, ">\n");
        }
        HCUDAERR(cudaMemcpy(gpu_clusters, clusters,
            sizeof(float)*nclusters*dims, cudaMemcpyHostToDevice));

        gpu_kmeans<<<nelems*dims/TPB + 1, TPB>>>(dims, i, limit, nclusters,
        gpu_clusters, nelems, gpu_data, gpu_elemClus,
        gpu_entropy, gpu_elemDis, gpu_movedDis);

        if (log) {
            // Copy back all data to log it
            copyToHost(nclusters*dims, clusters, gpu_clusters, nelems, elemClus,
            gpu_elemClus, elemDis, gpu_elemDis, epochs, entropy, gpu_entropy);

            logKmeansEpoch(log, dims, nclusters, clusters, nelems, data,
            elemClus, entropy[i], elemDis);
        } else {
            // TODO: Only copy the elemClus and entropy to calculate best cluster
            HCUDAERR(cudaMemcpy(entropy, gpu_entropy, sizeof(float)*epochs,
                cudaMemcpyDeviceToHost));
        }

        if (entropy[i] < bestEntropy) {
            HCUDAERR(cudaMemcpy(elemClus, gpu_elemDis,
                sizeof(int)*nelems, cudaMemcpyDeviceToHost));
            bestEntropy = entropy[i];
        }
    }

    free(clusters); free(entropy); free(elemDis);

    cudaFree(gpu_data); cudaFree(gpu_entropy);
    cudaFree(gpu_elemDis); cudaFree(gpu_movedDis);
//    cudaFree(gpu_clusters); cudaFree(gpu_elemClus);
    return elemClus;
}
#endif
