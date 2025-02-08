#include <iostream>
#include <array>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cuda.h>
#include <cudnn.h>

#define CHECK_CUDA(status) \
    do \
    { \
        if (status != cudaSuccess) \
        { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUDNN(status) \
    do \
    { \
        if (status != CUDNN_STATUS_SUCCESS) \
        { \
            std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template<typename T,cudnnDataType_t D>
void cudnn_conv2d(T* i1,int* d1,T* k2,int* d2,int* s3,int* p4,T* o5,int* d5)
{
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t xDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, D, d1[0], d1[3], d1[1], d1[2]));

    cudnnFilterDescriptor_t wDesc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, D, CUDNN_TENSOR_NHWC, d2[3], d2[2], d2[0], d2[1]));

    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, p4[2], p4[4], s3[1], s3[2], 1, 1, CUDNN_CROSS_CORRELATION, D));

    cudnnTensorDescriptor_t yDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    int N, C, H, W;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &N, &C, &H, &W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, D, N, C, H, W));

    T *dx, *dw, *dy;
    cudaMalloc(&dx, d1[0] * d1[1] * d1[2] * d1[3] * sizeof(T));
    cudaMalloc(&dw, d2[0] * d2[1] * d2[2] * d2[3] * sizeof(T));
    cudaMalloc(&dy, N * C * H * W * sizeof(T));

    Eigen::array<int, 4> n; for (int i=0;i < 4;i++) n[i] = d2[i];
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> k_(k2, n);
    Eigen::array<ptrdiff_t, 4> f = {3,0,1,2};
    Eigen::Tensor<T, 4, Eigen::RowMajor> k = k_.shuffle(f);

    CHECK_CUDA(cudaMemcpy(dx, i1, d1[0] * d1[1] * d1[2] * d1[3] * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw, k.data(), d2[0] * d2[1] * d2[2] * d2[3] * sizeof(T), cudaMemcpyHostToDevice));

    cudnnConvolutionFwdAlgo_t algo;
    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &workspaceSize));

    void *workspace;
    cudaMalloc(&workspace, workspaceSize);

    T alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, dx, wDesc, dw, convDesc, algo, workspace, workspaceSize, &beta, yDesc, dy));
    CHECK_CUDA(cudaMemcpy(o5, dy, N * C * H * W * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroy(handle);
}

extern "C" void pg_cudnn_convolve(int oid,void* i1,int n1,int* d1,void* k2,int* d2,int* s3,int* p4,void* o5,int* d5)
{
    if (oid == 700)
    {
        if (n1 == 4)
            cudnn_conv2d<float, CUDNN_DATA_FLOAT>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) o5, d5);
    }
    else if (oid == 701)
    {
        if (n1 == 4)
            cudnn_conv2d<double, CUDNN_DATA_DOUBLE>((double*) i1, d1, (double*) k2, d2, s3, p4, (double*) o5, d5);
    }
}
