#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TPB 512
#define NLIMIT 100

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
        dist = normf(dims, clusters + cPos);

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

    for (int i = 0; i < dims; i++) {
        avg = 0;
        for (int j = 0; j < nelems; j++)
            if (elemClus[j] == clusID)
                avg += elems[dims*j + i]/nelems;
        dis += fabsf(clusters[cPos+1] - avg);
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
void kmeans(const int dims, const int epochs, const int limit,
const int nclusters, float *clusters, const int nelems, float *elems,
int *elemClus, float *entropy) {
    // shared elemDis
    __shared__ float elemDis[nelems];
    // shared distance moved
    __shared__ float movedDis[nclusters];
    // shared best clusters
    __shared__ float bestClusters[nclusters*dims];

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    for (int i = 0; i < epochs; i++) {
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
    cv::Mat og, src, dst;
    int size, channels, nelems;
    int nclusters, epochs, limit;
    float *dstRaw;

    float *gpu_src, *gpu_dst, *gpu_clusters, *gpu_entropy;
    int *gpu_elemClus;

    if (argc != 5) {
        fprintf(
            stderr, 
            "usage: %s <image path> <n clusters> <epochs> <result file>\n",
            argv[0]
        );
        return -1;
    }

    // TODO: Scan number of clusters

    // TODO: Scan number of epochs

    // TODO: Scan limit

    // Scan image and convert it to float
    fprintf(stdout, "Reading image...\n");
    og = cv::imread(argv[1], cv::IMREAD_COLOR);
    og.convertTo(src, CV_32F);
    nelems = src.rows * src.cols;
    channels = src.channels();
    size = src.rows * src.cols * og.channels();

    // Allocate CPU
    fprintf(stdout, "Allocating memory in CPU...\n");
    dstRaw = (float *)malloc(sizeof(float)*size);

    // Copy to gpu
    fprintf(stdout, "Allocating memory in GPU...\n");
    cudaMalloc((void**) &gpu_src, sizeof(float)*size);
    cudaMalloc((void**) &gpu_clusters, sizeof(float)*nclusters*channels);
    cudaMalloc((void**) &gpu_elemClus, sizeof(int)*nelems);
    cudaMalloc((void**) &gpu_entropy, sizeof(float)*epochs);
    cudaMalloc((void**) &gpu_dst, sizeof(float)*size);

    fprintf(stdout, "Uploading image to GPU...\n");
    cudaMemcpy(gpu_src, src.data, sizeof(float)*size, cudaMemcpyHostToDevice);

    // Call kmeans
    fprintf(stdout, "Applying kmeans...\n");
    kmeans<<<nelems/TPB + 1, TPB>>>(channels, epochs, limit, nclusters,
    gpu_clusters, nelems, gpu_src, gpu_elemClus, gpu_entropy);

    // Call modified image
    fprintf(stdout, "Applying colors...\n");
    colorClusters<<<nelems/TPB + 1, TPB>>>(channels, gpu_clusters, nelems,
    gpu_elemClus, gpu_dst);

    // Copy processed data to CPU
    cudaMemcpy(dstRaw, gpu_dst, sizeof(float)*size, cudaMemcpyDeviceToHost);

    // Convert result to opencv
    fprintf(stdout, "Obtaining image from GPU...\n");
    dst = cv::Mat(src.rows, src.cols, src.type(), dstRaw, cv::Mat::AUTO_STEP);

    // Display images
    fprintf(stdout, "Displaying images...\n");
//    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Original", src);
//    cv::namedWindow("GrayScale", cv::WINDOW_AUTOSIZE);
//    cv::imshow("GrayScale", dst);
//    cv::waitKey(0);

    cudaFree(gpu_src); cudaFree(gpu_dst); cudaFree(gpu_clusters);
    cudaFree(gpu_elemClus); cudaFree(gpu_entropy);
    free(dstRaw);
    return 0;
}
