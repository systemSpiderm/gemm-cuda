#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

double get_random_double(double lower, double upper);

void initialize_matrix(double *matrix, int M, int N);

int main(int argc, char* argv[]) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    int M, N, K;

    if (argc != 4) {
        fprintf(stderr, "The program %s didnot get enough parameters, please enter M, N, K\n", argv[0]);
        exit(1);
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t size_A = M * N * sizeof(double);
    size_t size_B = N * K * sizeof(double);
    size_t size_C = M * K * sizeof(double);


    double* A = (double*)malloc(size_A);
    double* B = (double*)malloc(size_B);
    double* C = (double*)malloc(size_C);
    double *d_A, *d_B, *d_C;

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 设置矩阵乘法参数
    const double alpha = 1.0f; // 标量 alpha
    const double beta = 0.0f;  // 标量 beta

    // C = alpha * A * B + beta * C
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &alpha, d_A,
                M, d_B, N, &beta, d_C, M);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 清理资源
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("cublasDgemm ends, Elapsed time: %f ms\n", elapsedTime);

    free(A);
    free(B);
    free(C);

    return 0;
}


double get_random_double(double lower, double upper) {
    int random_int = rand();
    return lower + (double)(random_int / (RAND_MAX + 1.0)) * (upper - lower);
}

void initialize_matrix(double *matrix, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix[i] = get_random_double(100.0, 100000);
    }
}