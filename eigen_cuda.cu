#include <iostream>
#include <array>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

template<typename T,int L,int M>
void cuda_convolve(T* i1,int* d1,T* k2,int* d2,int* s3,int* p4,T* o5,int* d5)
{
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, M> pd;
    pd[0] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    if (p4 == NULL)
        for (int i=1;i < M;i++)
            pd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    else
        for (int i=1;i < M;i++)
            pd[i] = std::make_pair((ptrdiff_t)p4[2*i],(ptrdiff_t)p4[2*i+1]);
    Eigen::array<ptrdiff_t, M-1> cd;
    for (int i=0;i < M-1;i++) cd[i] = i+1;
    Eigen::array<ptrdiff_t, M> st;
    if (s3 == NULL)
        for (int i=0;i < M;i++) st[i] = 1;
    else
        for (int i=0;i < M;i++) st[i] = s3[i];
    std::size_t ib, kb, ub, cb;
    ib = kb = ub = cb =sizeof(T);
    Eigen::array<int, M> m, n, x;
    for (int i=0;i < M;i++)
    {
        ib *= d1[i];m[i] = d1[i];
        kb *= d2[i];n[i] = d2[i];
        ub *= d5[i];x[i] = d5[i];
    }
    ub /= d5[M-1];x[M-1] = 1;
    Eigen::array<int, M-1> y;
    for (int i=0;i < M-1;i++)
    {
        cb *= d2[i];y[i] = d2[i];
    }
    T* di; T* dk; T* du; T* dc; T* hp = o5;
    gpuMalloc((void**)(&di), ib);
    gpuMalloc((void**)(&dk), kb);
    gpuMalloc((void**)(&du), ub);
    gpuMalloc((void**)(&dc), cb);
    gpuMemcpy(di, i1, ib, gpuMemcpyHostToDevice);
    gpuMemcpy(dk, k2, kb, gpuMemcpyHostToDevice);
    Eigen::GpuStreamDevice stream;
    Eigen::GpuDevice gpu(&stream);
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gi(di, m);
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gk(dk, n);
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gu(du, x);
    Eigen::TensorMap<Eigen::Tensor<T,M-1,L>> gc(dc, y);
    for (int i = 0; i < d5[M-1]; i++)
    {
        gc.device(gpu) = gk.template chip<M-1>(i);
        gu.device(gpu) = gi.pad(pd).convolve(gc, cd).stride(st);
        gpuMemcpyAsync(hp, du, ub, gpuMemcpyDeviceToHost, gpu.stream());
        hp += ub/sizeof(T);
    }
    gpuStreamSynchronize(gpu.stream());
    gpuFree(di); gpuFree(dk); gpuFree(du);
    Eigen::array<int, M> z;z[0] = d5[M-1];
    for (int i=1;i < M;i++) z[i] = x[i-1];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> ht(o5, z);
    Eigen::array<int, M> f; f[M-1] = 0;
    for (int i=0;i < M-1;i++) f[i] = i+1;
    Eigen::Tensor<T, M, L> cv = ht.shuffle(f);
    std::copy(cv.data(), cv.data() + cv.size(), o5);
}

extern "C" void pg_cuda_convolve(int oid,void* i1,int n1,int* d1,void* k2,int* d2,int* s3,int* p4,void* o5,int* d5)
{
    if (oid == 700)
    {
        if (n1 == 3)
            cuda_convolve<float, Eigen::RowMajor, 3>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) o5, d5);
        else if (n1 == 4)
            cuda_convolve<float, Eigen::RowMajor, 4>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) o5, d5);
    }
}
