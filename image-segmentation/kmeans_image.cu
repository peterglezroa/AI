#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TPB 512
#define NLIMIT 100

#define LOGS 1
#define MAXDISPLAY 100

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
    float frand = curand_uniform(&(state[clusID]));
    frand *= nelems-1;
    int ePos = (int)truncf(frand);
    ePos *= dims;
    for (int i = 0; i < dims; i++)
        clusters[cPos+i] = elems[ePos+1];
}

/* Function to group the elements in its nearest cluster.
 * Returns the distance to said cluster. */
__device__
float grouping(const int elemID, const int dims, const int nclusters,
const float *clusters, const float *elems, int *elemClus) {
    float dist, elemDist, cPos;

    // Start by considering as part of the first cluster
    elemClus[elemID] = 0;
    elemDist = rnormf(dims, clusters);
    for (int i = 1; i < nclusters; i++) {
        // Calculate the position of the cluster in the linearized matrix
        cPos = i*dims;

        // Calculate distance
        dist = normf(dims, &clusters[(int)cPos]);

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

__global__
void colorClusters(const int dims, float *clusters, const int nelems,
int *elemClus, float *dst) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x), pos;

    if (tid < nelems) {
        pos = tid*dims;
        for (int i = 0; i < dims; i++)
            dst[pos+i] = clusters[dims*elemClus[tid]+i];
    }
}

int main(int argc, char *argv[]) {
    curandState *gpu_state;
    cv::Mat og, tmp, dst, finalDst;
    int size, channels, nelems;
    int nclusters, epochs, limit;
    float *dstRaw, *src;

    float *gpu_src, *gpu_dst, *gpu_clusters, *gpu_entropy;
    int *gpu_elemClus;
    float *gpu_elemDis, *gpu_movedDis, *gpu_bC;

    if (argc != 6) {
        fprintf(
            stderr, 
            "usage: %s <image> <n clusters> <epochs> <nlimit> <result img>\n",
            argv[0]
        );
        return -1;
    }

    // Scan number of clusters
    if ( (nclusters = atoi(argv[2])) == 0) {
        fprintf(stderr, "%s is not a valid number for nclusters\n", argv[2]);
        return -2;
    }

    // Scan number of epochs
    if ( (epochs = atoi(argv[3])) == 0) {
        fprintf(stderr, "%s is not a valid number for epochs\n", argv[3]);
        return -3;
    }

    // Scan limit
    if ( (limit = atoi(argv[4])) == 0) {
        fprintf(stderr, "%s is not a valid number for nlimit\n", argv[4]);
        return -4;
    }

    // Scan image and convert it to float
    fprintf(stdout, "Reading image...\n");
    og = cv::imread(argv[1], cv::IMREAD_COLOR);
    og.convertTo(tmp, CV_32F);
    src = (float*)tmp.data;
    nelems = og.rows * og.cols;
    channels = og.channels();
    size = og.rows * og.cols * og.channels();

    // Allocate CPU
    fprintf(stdout, "Allocating memory in CPU...\n");
    dstRaw = (float *)malloc(sizeof(float)*size);

    // Copy to gpu
    fprintf(stdout, "Allocating memory in GPU...\n");
    cudaMalloc(&gpu_state, sizeof(curandState));
    cudaMalloc((void**) &gpu_src, sizeof(float)*size);
    cudaMalloc((void**) &gpu_clusters, sizeof(float)*nclusters*channels);
    cudaMalloc((void**) &gpu_elemClus, sizeof(int)*nelems);
    cudaMalloc((void**) &gpu_entropy, sizeof(float)*epochs);
    cudaMalloc((void**) &gpu_dst, sizeof(float)*size);

    // Element distance to its cluster
    cudaMalloc((void**) &gpu_elemDis, sizeof(float)*nelems);
    // Distance that the cluster moved
    cudaMalloc((void**) &gpu_movedDis, sizeof(float)*nclusters);
    // Saved best performing clusters
    cudaMalloc((void**) &gpu_bC, sizeof(float)*nclusters*channels);

    fprintf(stdout, "Uploading image to GPU...\n");
    cudaMemcpy(gpu_src, src, sizeof(float)*size, cudaMemcpyHostToDevice);

    // Setup kernel
    fprintf(stdout, "Setting up GPU kernel state...\n");
    setupKernel<<<nelems/TPB+1, TPB>>>(gpu_state);

    // Call kmeans
    fprintf(stdout, "Applying kmeans...\n");
    kmeans<<<nelems/TPB + 1, TPB>>>(gpu_state, channels, epochs, limit, nclusters,
    gpu_clusters, nelems, gpu_src, gpu_elemClus, gpu_entropy,
    gpu_elemDis, gpu_movedDis, gpu_bC);

    // Print logs
    if(LOGS) {
        float *clusters = (float *)malloc(sizeof(float)*nclusters*channels);
        cudaMemcpy(clusters, gpu_clusters, sizeof(float)*nclusters*channels,
            cudaMemcpyDeviceToHost);

        float *entropy = (float *)malloc(sizeof(float)*epochs);
        cudaMemcpy(entropy, gpu_entropy, sizeof(float)*epochs,
            cudaMemcpyDeviceToHost);

        float *elemDis = (float *)malloc(sizeof(float)*nelems);
        cudaMemcpy(elemDis, gpu_elemDis, sizeof(float)*nelems,
            cudaMemcpyDeviceToHost);

        int *elemClus = (int *)malloc(sizeof(float)*nelems);
        cudaMemcpy(elemClus, gpu_elemClus, sizeof(int)*nelems,
            cudaMemcpyDeviceToHost);

        fprintf(stdout, "Resulting clusters:\n");
        // Log clusters
        for (int i = 0; i < nclusters; i++) {
            fprintf(stdout, "\t#%i: ", i);
            for (int j = 0; j < channels; j++)
                fprintf(stdout, "%.1f ", clusters[i*channels+j]);
            fprintf(stdout, "\n");
        }

        fprintf(stdout, "\nElements:\n");
        // Log elements and its distance
        for (int i = 0; i < nelems && i < MAXDISPLAY; i++) {
            fprintf(stdout, "\t#%i: ", i);
            for (int j = 0; j < channels; j++)
                fprintf(stdout, "%.1f ", src[i*channels+j]);
            fprintf(stdout, "Cluster (%i, %.1f)\n", elemClus[i], elemDis[i]);
        }

        fprintf(stdout, "\nEntropies:\n");
        for (int i = 0; i < epochs; i++)
            fprintf(stdout, "\t%.3f", entropy[i]);

        free(clusters); free(entropy); free(elemDis); free(elemClus);
    }

    /*
    // Call modified image
    fprintf(stdout, "Applying colors...\n");
    colorClusters<<<nelems/TPB + 1, TPB>>>(channels, gpu_clusters, nelems,
    gpu_elemClus, gpu_dst);

    // Copy processed data to CPU
    cudaMemcpy(dstRaw, gpu_dst, sizeof(float)*size, cudaMemcpyDeviceToHost);

    // Convert result to opencv
    fprintf(stdout, "Obtaining image from GPU...\n");
    dst = cv::Mat(og.rows, og.cols, og.type(), dstRaw, cv::Mat::AUTO_STEP);
    dst.convertTo(finalDst, CV_8U);

    // Display images
    fprintf(stdout, "Displaying images...\n");
//    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Original", og);
//    cv::namedWindow("GrayScale", cv::WINDOW_AUTOSIZE);
//    cv::imshow("GrayScale", dst);
//    cv::waitKey(0);
    */

    cudaFree(gpu_src); cudaFree(gpu_dst); cudaFree(gpu_clusters);
    cudaFree(gpu_elemClus); cudaFree(gpu_entropy);
    free(dstRaw);
    return 0;
}
