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

#include "cuda_kmeans.cuh"

#define TPB 512
#define NLIMIT 100

#define LOGS 1
#define MAXDISPLAY 100

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

__global__
void test(curandState *state, const int dims, const int epochs, const int limit,
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

//            __syncthreads();

        }
        // TODO: copy to bestCluster
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
    fprintf(stdout, "Image %s (%ix%ix%i)", argv[1], og.rows, og.cols,
        og.channels());
    fprintf(stdout, "Converting image to CV_32F...\n");
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
    HCUDAERR(cudaMalloc(&gpu_state, sizeof(curandState)));
    HCUDAERR(cudaMalloc((void**) &gpu_src, sizeof(float)*size));
    HCUDAERR(cudaMalloc((void**) &gpu_clusters, sizeof(float)*nclusters*channels));
    HCUDAERR(cudaMalloc((void**) &gpu_elemClus, sizeof(int)*nelems));
    HCUDAERR(cudaMalloc((void**) &gpu_entropy, sizeof(float)*epochs));
    HCUDAERR(cudaMalloc((void**) &gpu_dst, sizeof(float)*size));

    // Element distance to its cluster
    HCUDAERR(cudaMalloc((void**) &gpu_elemDis, sizeof(float)*nelems));
    // Distance that the cluster moved
    HCUDAERR(cudaMalloc((void**) &gpu_movedDis, sizeof(float)*nclusters));
    // Saved best performing clusters
    HCUDAERR(cudaMalloc((void**) &gpu_bC, sizeof(float)*nclusters*channels));

    fprintf(stdout, "Uploading image to GPU...\n");
    HCUDAERR(cudaMemcpy(gpu_src, src, sizeof(float)*size, cudaMemcpyHostToDevice));

    // Setup kernel
    fprintf(stdout, "Setting up GPU kernel state...\n");
    setupKernel<<<1, 1>>>(gpu_state);

    // Call kmeans
    fprintf(stdout, "Applying kmeans...\n");
    /*
    kmeans<<<nelems/TPB + 1, TPB>>>(gpu_state, channels, epochs, limit, nclusters,
    gpu_clusters, nelems, gpu_src, gpu_elemClus, gpu_entropy,
    gpu_elemDis, gpu_movedDis, gpu_bC);
    */
    float *clusters = (float *)malloc(sizeof(float)*nclusters*channels);
    for (int i = 0; i < nclusters*channels; i++) { clusters[i] = i; }
    HCUDAERR(cudaMemcpy(gpu_clusters, clusters, sizeof(float)*nclusters*channels,
        cudaMemcpyHostToDevice));

    test<<<nclusters*channels/TPB + 1, TPB>>>(gpu_state, channels, epochs, limit, nclusters,
    gpu_clusters, nelems, gpu_src, gpu_elemClus, gpu_entropy,
    gpu_elemDis, gpu_movedDis, gpu_bC);

    // Print logs
    if(LOGS) {
        float *clusters = (float *)malloc(sizeof(float)*nclusters*channels);
        cudaMemcpy(clusters, gpu_clusters, sizeof(float)*nclusters*channels,
            cudaMemcpyDeviceToHost);

        fprintf(stdout, "Resulting clusters:\n");
        // Log clusters
        for (int i = 0; i < nclusters; i++) {
            fprintf(stdout, "\t#%i: ", i);
            for (int j = 0; j < channels; j++)
                fprintf(stdout, "%.1f ", clusters[i*channels+j]);
            fprintf(stdout, "\n");
        }

        float *entropy = (float *)malloc(sizeof(float)*epochs);
        cudaMemcpy(entropy, gpu_entropy, sizeof(float)*epochs,
            cudaMemcpyDeviceToHost);
        fprintf(stdout, "\nEntropies:\n");
        for (int i = 0; i < epochs; i++)
            fprintf(stdout, "\t%.3f", entropy[i]);
        fprintf(stdout, "\n");

        float *elemDis = (float *)malloc(sizeof(float)*nelems);
        cudaMemcpy(elemDis, gpu_elemDis, sizeof(float)*nelems,
            cudaMemcpyDeviceToHost);

        int *elemClus = (int *)malloc(sizeof(float)*nelems);
        cudaMemcpy(elemClus, gpu_elemClus, sizeof(int)*nelems,
            cudaMemcpyDeviceToHost);
        fprintf(stdout, "\nElements:\n");
        // Log elements and its distance
        for (int i = 0; i < nelems && i < MAXDISPLAY; i++) {
            fprintf(stdout, "\t#%i: ", i);
            for (int j = 0; j < channels; j++)
                fprintf(stdout, "%.1f ", src[i*channels+j]);
            fprintf(stdout, "Cluster (%i, %.1f)\n", elemClus[i], elemDis[i]);
        }

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
