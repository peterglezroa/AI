#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

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
        // TODO: copy to bestCluster
    }
}

int main(int argc, char *argv[]) {
    cv::Mat og, tmp, dst, finalDst;
    int size, channels, nelems, *elemClus;
    int nclusters, epochs, limit;
    float *dstRaw, *src;

    srand(time(NULL));

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
//    dstRaw = (float *)malloc(sizeof(float)*size);

    // Calling kmeans
    elemClus = kmeans(channels, epochs, limit, nclusters, nelems, src, stdout);

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

//    free(dstRaw);
    free(elemClus);
    return 0;
}
