#include <iostream>
#include <array>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename T,int L,int M>
void tensor_reduce(int fn,T* in,int* d1,T* out)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(in, m);
    if (fn == 1)
    {
        Eigen::Tensor<T, 0, L> y = x.sum();
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 2)
    {
        Eigen::Tensor<T, 0, L> y = x.mean();
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 3)
    {
        Eigen::Tensor<T, 0, L> y = x.prod();
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 4)
    {
        Eigen::Tensor<T, 0, L> y = x.maximum();
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 5)
    {
        Eigen::Tensor<T, 0, L> y = x.minimum();
        std::copy(y.data(), y.data() + y.size(), out);
    }
}

template<typename T,int L,int M,int N>
void tensor_reduce(int fn,T* in,int* d1,T* out,int* d2)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(in, m);
    Eigen::array<int, N> r;
    for (int i=0;i < N;i++) r[i] = d2[i];
    if (fn == 1)
    {
        Eigen::Tensor<T, M-N, L> y = x.sum(r);
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 2)
    {
        Eigen::Tensor<T, M-N, L> y = x.mean(r);
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 3)
    {
        Eigen::Tensor<T, M-N, L> y = x.prod(r);
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 4)
    {
        Eigen::Tensor<T, M-N, L> y = x.maximum(r);
        std::copy(y.data(), y.data() + y.size(), out);
    }
    else if (fn == 5)
    {
        Eigen::Tensor<T, M-N, L> y = x.minimum(r);
        std::copy(y.data(), y.data() + y.size(), out);
    }
}

extern "C" void pg_tensor_reduce(int oid,int fn,char* in,int n1,int* d1,void* out,int n2,int* d2)
{
    if (oid == 700)
    {
        if (n1 == 1)
        {
            tensor_reduce<float, Eigen::RowMajor, 1>(fn, (float *)in, d1, (float *)out);
            return;
        }
        else if (n1 == 2)
        {
            if (n2 == 1)
                tensor_reduce<float, Eigen::RowMajor, 2, 1>(fn, (float *)in, d1, (float *)out, d2);
            else
                tensor_reduce<float, Eigen::RowMajor, 2>(fn, (float *)in, d1, (float *)out);
            return;
        }
        else if (n1 == 3)
        {
            if (n2 == 1)
                tensor_reduce<float, Eigen::RowMajor, 3, 1>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 2)
                tensor_reduce<float, Eigen::RowMajor, 3, 2>(fn, (float *)in, d1, (float *)out, d2);
            else
                tensor_reduce<float, Eigen::RowMajor, 3>(fn, (float *)in, d1, (float *)out);
            return;
        }
        else if (n1 == 4)
        {
            if (n2 == 1)
                tensor_reduce<float, Eigen::RowMajor, 4, 1>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 2)
                tensor_reduce<float, Eigen::RowMajor, 4, 2>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 3)
                tensor_reduce<float, Eigen::RowMajor, 4, 3>(fn, (float *)in, d1, (float *)out, d2);
            else
                tensor_reduce<float, Eigen::RowMajor, 4>(fn, (float *)in, d1, (float *)out);
            return;
        }
        else if (n1 == 5)
        {
            if (n2 == 1)
                tensor_reduce<float, Eigen::RowMajor, 5, 1>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 2)
                tensor_reduce<float, Eigen::RowMajor, 5, 2>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 3)
                tensor_reduce<float, Eigen::RowMajor, 5, 3>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 4)
                tensor_reduce<float, Eigen::RowMajor, 5, 4>(fn, (float *)in, d1, (float *)out, d2);
            else
                tensor_reduce<float, Eigen::RowMajor, 5>(fn, (float *)in, d1, (float *)out);
            return;
        }
        else if (n1 == 6)
        {
            if (n2 == 1)
                tensor_reduce<float, Eigen::RowMajor, 6, 1>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 2)
                tensor_reduce<float, Eigen::RowMajor, 6, 2>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 3)
                tensor_reduce<float, Eigen::RowMajor, 6, 3>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 4)
                tensor_reduce<float, Eigen::RowMajor, 6, 4>(fn, (float *)in, d1, (float *)out, d2);
            else if (n2 == 5)
                tensor_reduce<float, Eigen::RowMajor, 6, 5>(fn, (float *)in, d1, (float *)out, d2);
            else
                tensor_reduce<float, Eigen::RowMajor, 6>(fn, (float *)in, d1, (float *)out);
            return;
        }
    }
    else if (oid == 701)
    {
        if (n1 == 1)
        {
            tensor_reduce<double, Eigen::RowMajor, 1>(fn, (double *)in, d1, (double *)out);
            return;
        }
        else if (n1 == 2)
        {
            if (n2 == 1)
                tensor_reduce<double, Eigen::RowMajor, 2, 1>(fn, (double *)in, d1, (double *)out, d2);
            else
                tensor_reduce<double, Eigen::RowMajor, 2>(fn, (double *)in, d1, (double *)out);
            return;
        }
        else if (n1 == 3)
        {
            if (n2 == 1)
                tensor_reduce<double, Eigen::RowMajor, 3, 1>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 2)
                tensor_reduce<double, Eigen::RowMajor, 3, 2>(fn, (double *)in, d1, (double *)out, d2);
            else
                tensor_reduce<double, Eigen::RowMajor, 3>(fn, (double *)in, d1, (double *)out);
            return;
        }
        else if (n1 == 4)
        {
            if (n2 == 1)
                tensor_reduce<double, Eigen::RowMajor, 4, 1>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 2)
                tensor_reduce<double, Eigen::RowMajor, 4, 2>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 3)
                tensor_reduce<double, Eigen::RowMajor, 4, 3>(fn, (double *)in, d1, (double *)out, d2);
            else
                tensor_reduce<double, Eigen::RowMajor, 4>(fn, (double *)in, d1, (double *)out);
            return;
        }
        else if (n1 == 5)
        {
            if (n2 == 1)
                tensor_reduce<double, Eigen::RowMajor, 5, 1>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 2)
                tensor_reduce<double, Eigen::RowMajor, 5, 2>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 3)
                tensor_reduce<double, Eigen::RowMajor, 5, 3>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 4)
                tensor_reduce<double, Eigen::RowMajor, 5, 4>(fn, (double *)in, d1, (double *)out, d2);
            else
                tensor_reduce<double, Eigen::RowMajor, 5>(fn, (double *)in, d1, (double *)out);
            return;
        }
        else if (n1 == 6)
        {
            if (n2 == 1)
                tensor_reduce<double, Eigen::RowMajor, 6, 1>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 2)
                tensor_reduce<double, Eigen::RowMajor, 6, 2>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 3)
                tensor_reduce<double, Eigen::RowMajor, 6, 3>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 4)
                tensor_reduce<double, Eigen::RowMajor, 6, 4>(fn, (double *)in, d1, (double *)out, d2);
            else if (n2 == 5)
                tensor_reduce<double, Eigen::RowMajor, 6, 5>(fn, (double *)in, d1, (double *)out, d2);
            else
                tensor_reduce<double, Eigen::RowMajor, 6>(fn, (double *)in, d1, (double *)out);
            return;
        }
    }
    else if (oid ==  21)
    {
        if (n1 == 1)
        {
            tensor_reduce<short, Eigen::RowMajor, 1>(fn, (short *)in, d1, (short *)out);
            return;
        }
        else if (n1 == 2)
        {
            if (n2 == 1)
                tensor_reduce<short, Eigen::RowMajor, 2, 1>(fn, (short *)in, d1, (short *)out, d2);
            else
                tensor_reduce<short, Eigen::RowMajor, 2>(fn, (short *)in, d1, (short *)out);
            return;
        }
        else if (n1 == 3)
        {
            if (n2 == 1)
                tensor_reduce<short, Eigen::RowMajor, 3, 1>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 2)
                tensor_reduce<short, Eigen::RowMajor, 3, 2>(fn, (short *)in, d1, (short *)out, d2);
            else
                tensor_reduce<short, Eigen::RowMajor, 3>(fn, (short *)in, d1, (short *)out);
            return;
        }
        else if (n1 == 4)
        {
            if (n2 == 1)
                tensor_reduce<short, Eigen::RowMajor, 4, 1>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 2)
                tensor_reduce<short, Eigen::RowMajor, 4, 2>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 3)
                tensor_reduce<short, Eigen::RowMajor, 4, 3>(fn, (short *)in, d1, (short *)out, d2);
            else
                tensor_reduce<short, Eigen::RowMajor, 4>(fn, (short *)in, d1, (short *)out);
            return;
        }
        else if (n1 == 5)
        {
            if (n2 == 1)
                tensor_reduce<short, Eigen::RowMajor, 5, 1>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 2)
                tensor_reduce<short, Eigen::RowMajor, 5, 2>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 3)
                tensor_reduce<short, Eigen::RowMajor, 5, 3>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 4)
                tensor_reduce<short, Eigen::RowMajor, 5, 4>(fn, (short *)in, d1, (short *)out, d2);
            else
                tensor_reduce<short, Eigen::RowMajor, 5>(fn, (short *)in, d1, (short *)out);
            return;
        }
        else if (n1 == 6)
        {
            if (n2 == 1)
                tensor_reduce<short, Eigen::RowMajor, 6, 1>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 2)
                tensor_reduce<short, Eigen::RowMajor, 6, 2>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 3)
                tensor_reduce<short, Eigen::RowMajor, 6, 3>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 4)
                tensor_reduce<short, Eigen::RowMajor, 6, 4>(fn, (short *)in, d1, (short *)out, d2);
            else if (n2 == 5)
                tensor_reduce<short, Eigen::RowMajor, 6, 5>(fn, (short *)in, d1, (short *)out, d2);
            else
                tensor_reduce<short, Eigen::RowMajor, 6>(fn, (short *)in, d1, (short *)out);
            return;
        }
    }
    else if (oid ==  23)
    {
        if (n1 == 1)
        {
            tensor_reduce<int, Eigen::RowMajor, 1>(fn, (int *)in, d1, (int *)out);
            return;
        }
        else if (n1 == 2)
        {
            if (n2 == 1)
                tensor_reduce<int, Eigen::RowMajor, 2, 1>(fn, (int *)in, d1, (int *)out, d2);
            else
                tensor_reduce<int, Eigen::RowMajor, 2>(fn, (int *)in, d1, (int *)out);
            return;
        }
        else if (n1 == 3)
        {
            if (n2 == 1)
                tensor_reduce<int, Eigen::RowMajor, 3, 1>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 2)
                tensor_reduce<int, Eigen::RowMajor, 3, 2>(fn, (int *)in, d1, (int *)out, d2);
            else
                tensor_reduce<int, Eigen::RowMajor, 3>(fn, (int *)in, d1, (int *)out);
            return;
        }
        else if (n1 == 4)
        {
            if (n2 == 1)
                tensor_reduce<int, Eigen::RowMajor, 4, 1>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 2)
                tensor_reduce<int, Eigen::RowMajor, 4, 2>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 3)
                tensor_reduce<int, Eigen::RowMajor, 4, 3>(fn, (int *)in, d1, (int *)out, d2);
            else
                tensor_reduce<int, Eigen::RowMajor, 4>(fn, (int *)in, d1, (int *)out);
            return;
        }
        else if (n1 == 5)
        {
            if (n2 == 1)
                tensor_reduce<int, Eigen::RowMajor, 5, 1>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 2)
                tensor_reduce<int, Eigen::RowMajor, 5, 2>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 3)
                tensor_reduce<int, Eigen::RowMajor, 5, 3>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 4)
                tensor_reduce<int, Eigen::RowMajor, 5, 4>(fn, (int *)in, d1, (int *)out, d2);
            else
                tensor_reduce<int, Eigen::RowMajor, 5>(fn, (int *)in, d1, (int *)out);
            return;
        }
        else if (n1 == 6)
        {
            if (n2 == 1)
                tensor_reduce<int, Eigen::RowMajor, 6, 1>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 2)
                tensor_reduce<int, Eigen::RowMajor, 6, 2>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 3)
                tensor_reduce<int, Eigen::RowMajor, 6, 3>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 4)
                tensor_reduce<int, Eigen::RowMajor, 6, 4>(fn, (int *)in, d1, (int *)out, d2);
            else if (n2 == 5)
                tensor_reduce<int, Eigen::RowMajor, 6, 5>(fn, (int *)in, d1, (int *)out, d2);
            else
                tensor_reduce<int, Eigen::RowMajor, 6>(fn, (int *)in, d1, (int *)out);
            return;
        }
    }
    else if (oid ==  20)
    {
        if (n1 == 1)
        {
            tensor_reduce<long, Eigen::RowMajor, 1>(fn, (long *)in, d1, (long *)out);
            return;
        }
        else if (n1 == 2)
        {
            if (n2 == 1)
                tensor_reduce<long, Eigen::RowMajor, 2, 1>(fn, (long *)in, d1, (long *)out, d2);
            else
                tensor_reduce<long, Eigen::RowMajor, 2>(fn, (long *)in, d1, (long *)out);
            return;
        }
        else if (n1 == 3)
        {
            if (n2 == 1)
                tensor_reduce<long, Eigen::RowMajor, 3, 1>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 2)
                tensor_reduce<long, Eigen::RowMajor, 3, 2>(fn, (long *)in, d1, (long *)out, d2);
            else
                tensor_reduce<long, Eigen::RowMajor, 3>(fn, (long *)in, d1, (long *)out);
            return;
        }
        else if (n1 == 4)
        {
            if (n2 == 1)
                tensor_reduce<long, Eigen::RowMajor, 4, 1>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 2)
                tensor_reduce<long, Eigen::RowMajor, 4, 2>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 3)
                tensor_reduce<long, Eigen::RowMajor, 4, 3>(fn, (long *)in, d1, (long *)out, d2);
            else
                tensor_reduce<long, Eigen::RowMajor, 4>(fn, (long *)in, d1, (long *)out);
            return;
        }
        else if (n1 == 5)
        {
            if (n2 == 1)
                tensor_reduce<long, Eigen::RowMajor, 5, 1>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 2)
                tensor_reduce<long, Eigen::RowMajor, 5, 2>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 3)
                tensor_reduce<long, Eigen::RowMajor, 5, 3>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 4)
                tensor_reduce<long, Eigen::RowMajor, 5, 4>(fn, (long *)in, d1, (long *)out, d2);
            else
                tensor_reduce<long, Eigen::RowMajor, 5>(fn, (long *)in, d1, (long *)out);
            return;
        }
        else if (n1 == 6)
        {
            if (n2 == 1)
                tensor_reduce<long, Eigen::RowMajor, 6, 1>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 2)
                tensor_reduce<long, Eigen::RowMajor, 6, 2>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 3)
                tensor_reduce<long, Eigen::RowMajor, 6, 3>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 4)
                tensor_reduce<long, Eigen::RowMajor, 6, 4>(fn, (long *)in, d1, (long *)out, d2);
            else if (n2 == 5)
                tensor_reduce<long, Eigen::RowMajor, 6, 5>(fn, (long *)in, d1, (long *)out, d2);
            else
                tensor_reduce<long, Eigen::RowMajor, 6>(fn, (long *)in, d1, (long *)out);
            return;
        }
    }
}

template<typename T,int L,int M,int R,int D>
void tensor_rfft(T* in,int* d1,T* out,int* d2)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(in, m);
    Eigen::array<int, M> f;
    for (int i=0;i < M;i++) f[i] = d2[i];
    Eigen::Tensor<std::complex<T>, M, L> y = x.template fft<R, D>(f);
    for (Eigen::DenseIndex i=0;i < y.size();i++)
    {
        out[2*i] = y(i).real();
        out[2*i+1] = y(i).imag();
    }
}

template<typename T,int L,int M,int R,int D>
void tensor_fft(std::complex<T>* in,int* d1,T* out,int* d2)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<std::complex<T>, M, L>> x(in, m);
    Eigen::array<int, M> f;
    for (int i=0;i < M;i++) f[i] = d2[i];
    Eigen::Tensor<std::complex<T>, M, L> y = x.template fft<R, D>(f);
    for (Eigen::DenseIndex i=0;i < y.size();i++)
    {
        out[2*i] = y(i).real();
        out[2*i+1] = y(i).imag();
    }
}

extern "C" void pg_tensor_fft(int oid,bool forward,char* in,int n1,int* d1,void* out,int n2,int* d2)
{
    if (forward)
    {
        if (n1 == n2)
        {
            if (oid == 700)
            {
                if (n1 == 1)
                    tensor_rfft<float, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_FORWARD>((float *)in, d1, (float *)out, d2);
                else if (n1 == 2)
                    tensor_rfft<float, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_FORWARD>((float *)in, d1, (float *)out, d2);
                else if (n1 == 3)
                    tensor_rfft<float, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_FORWARD>((float *)in, d1, (float *)out, d2);
                else if (n1 == 4)
                    tensor_rfft<float, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_FORWARD>((float *)in, d1, (float *)out, d2);
                else if (n1 == 5)
                    tensor_rfft<float, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_FORWARD>((float *)in, d1, (float *)out, d2);
            }
            else if (oid == 701)
            {
                if (n1 == 1)
                    tensor_rfft<double, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_FORWARD>((double *)in, d1, (double *)out, d2);
                else if (n1 == 2)
                    tensor_rfft<double, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_FORWARD>((double *)in, d1, (double *)out, d2);
                else if (n1 == 3)
                    tensor_rfft<double, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_FORWARD>((double *)in, d1, (double *)out, d2);
                else if (n1 == 4)
                    tensor_rfft<double, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_FORWARD>((double *)in, d1, (double *)out, d2);
                else if (n1 == 5)
                    tensor_rfft<double, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_FORWARD>((double *)in, d1, (double *)out, d2);
            }
        }
        else if (n1 == n2 + 1)
        {
            if (oid == 700)
            {
                if (n1 == 2)
                    tensor_fft<float, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 3)
                    tensor_fft<float, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 4)
                    tensor_fft<float, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 5)
                    tensor_fft<float, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 6)
                    tensor_fft<float, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<float> *)in, d1, (float *)out, d2);
            }
            else if (oid == 701)
            {
                if (n1 == 2)
                    tensor_fft<double, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 3)
                    tensor_fft<double, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 4)
                    tensor_fft<double, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 5)
                    tensor_fft<double, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 6)
                    tensor_fft<double, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_FORWARD>((std::complex<double> *)in, d1, (double *)out, d2);
            }
        }
    }
    else
    {
        if (n1 == n2)
        {
            if (oid == 700)
            {
                if (n1 == 1)
                    tensor_rfft<float, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_REVERSE>((float *)in, d1, (float *)out, d2);
                else if (n1 == 2)
                    tensor_rfft<float, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_REVERSE>((float *)in, d1, (float *)out, d2);
                else if (n1 == 3)
                    tensor_rfft<float, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_REVERSE>((float *)in, d1, (float *)out, d2);
                else if (n1 == 4)
                    tensor_rfft<float, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_REVERSE>((float *)in, d1, (float *)out, d2);
                else if (n1 == 5)
                    tensor_rfft<float, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_REVERSE>((float *)in, d1, (float *)out, d2);
            }
            else if (oid == 701)
            {
                if (n1 == 1)
                    tensor_rfft<double, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_REVERSE>((double *)in, d1, (double *)out, d2);
                else if (n1 == 2)
                    tensor_rfft<double, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_REVERSE>((double *)in, d1, (double *)out, d2);
                else if (n1 == 3)
                    tensor_rfft<double, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_REVERSE>((double *)in, d1, (double *)out, d2);
                else if (n1 == 4)
                    tensor_rfft<double, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_REVERSE>((double *)in, d1, (double *)out, d2);
                else if (n1 == 5)
                    tensor_rfft<double, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_REVERSE>((double *)in, d1, (double *)out, d2);
            }
        }
        else if (n1 == n2 + 1)
        {
            if (oid == 700)
            {
                if (n1 == 2)
                    tensor_fft<float, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 3)
                    tensor_fft<float, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 4)
                    tensor_fft<float, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 5)
                    tensor_fft<float, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<float> *)in, d1, (float *)out, d2);
                else if (n1 == 6)
                    tensor_fft<float, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<float> *)in, d1, (float *)out, d2);
            }
            else if (oid == 701)
            {
                if (n1 == 2)
                    tensor_fft<double, Eigen::RowMajor, 1, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 3)
                    tensor_fft<double, Eigen::RowMajor, 2, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 4)
                    tensor_fft<double, Eigen::RowMajor, 3, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 5)
                    tensor_fft<double, Eigen::RowMajor, 4, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<double> *)in, d1, (double *)out, d2);
                else if (n1 == 6)
                    tensor_fft<double, Eigen::RowMajor, 5, Eigen::BothParts, Eigen::FFT_REVERSE>((std::complex<double> *)in, d1, (double *)out, d2);
            }
        }
    }
}

extern "C" void pg_tensor_random(int fn,int c1,double* out,double a1,double b1,int s1)
{
    unsigned int seed;
    if (s1 < 0)
    {
        std::random_device rd;
        seed = rd();
    }
    else
        seed = s1;
    std::mt19937 gen(seed);
    if (fn == 1)
    {
        std::normal_distribution<double> nd(a1, b1);
        for (int i = 0; i < c1; ++i)
            out[i] = nd(gen);
    }
    else if (fn == 2)
    {
        double lb = a1 - 2 * b1, ub = a1 + 2 * b1;
        std::normal_distribution<double> td(a1, b1);
        for (int i = 0; i < c1; ++i)
        {
            double tmp;
            do { tmp = td(gen); } while (tmp < lb || tmp > ub);
            out[i] = tmp;
        }
    }
    else if (fn == 3)
    {
        std::uniform_real_distribution<double> ud(a1, b1);
        for (int i = 0; i < c1; ++i)
            out[i] = ud(gen);
    }
    else if (fn == 4)
    {
        std::gamma_distribution<double> gd(a1, b1);
        for (int i = 0; i < c1; ++i)
            out[i] = gd(gen);
    }
}

extern "C" void pg_tensor_shuffle(int oid,int s1, int c1,void* out)
{
    std::default_random_engine r{std::random_device{}()};
    if (oid == 700)
    {
        for (int i = 0;(i * s1) < c1;i++)
        {
            std::shuffle((float*)out, (float*)out + s1, r);
            out = (float*)out + s1;
        }
    }
    else if (oid == 701)
    {
        for (int i = 0;(i * s1) < c1;i++)
        {
            std::shuffle((double*)out, (double*)out + s1, r);
            out = (double*)out + s1;
        }
    }
    else if (oid ==  21)
    {
        for (int i = 0;(i * s1) < c1;i++)
        {
            std::shuffle((short*)out, (short*)out + s1, r);
            out = (short*)out + s1;
        }
    }
    else if (oid ==  23)
    {
        for (int i = 0;(i * s1) < c1;i++)
        {
            std::shuffle((int*)out, (int*)out + s1, r);
            out = (int*)out + s1;
        }
    }
    else if (oid ==  20)
    {
        for (int i = 0;(i * s1) < c1;i++)
        {
            std::shuffle((long*)out, (long*)out + s1, r);
            out = (long*)out + s1;
        }
    }
}

template<typename T,int L>
void tensor_binaryop(int fn,int m,T* a,T* b)
{
    Eigen::Tensor<T, 1, L> n;
    Eigen::TensorMap<Eigen::Tensor<T, 1, L>> x(a, m);
    Eigen::TensorMap<Eigen::Tensor<T, 1, L>> y(b, m);
    switch (fn)
    {
    case 1:
        x += y;
        break;
    case 2:
        x -= y;
        break;
    case 3:
        x *= y;
        break;
    case 4:
        x /= y;
        break;
    case 5:
        n = (x == y).template cast<T>();
        break;
    case 6:
        n = (x != y).template cast<T>();
        break;
    case 7:
        n = (x  < y).template cast<T>();
        break;
    case 8:
        n = (x <= y).template cast<T>();
        break;
    case 9:
        n = (x  > y).template cast<T>();
        break;
    case 10:
        n = (x >= y).template cast<T>();
        break;
    default:
        break;
    }
    if (n.dimension(0) == m)
        std::copy(n.data(), n.data() + n.size(), a);
}

extern "C" void pg_tensor_binaryop(int oid,int fn,int c1,void* a1,void* a2)
{
    if (oid == 700)
        tensor_binaryop<float, Eigen::RowMajor>(fn, c1, (float*) a1, (float*) a2);
    else if (oid == 701)
        tensor_binaryop<double, Eigen::RowMajor>(fn, c1, (double*) a1, (double*) a2);
    else if (oid ==  21)
        tensor_binaryop<short, Eigen::RowMajor>(fn, c1, (short*) a1, (short*) a2);
    else if (oid ==  23)
        tensor_binaryop<int, Eigen::RowMajor>(fn, c1, (int*) a1, (int*) a2);
    else if (oid ==  20)
        tensor_binaryop<long, Eigen::RowMajor>(fn, c1, (long*) a1, (long*) a2);
}

template<typename T,int L>
void tensor_unaryop(int fn,int m,T* a)
{
    Eigen::TensorMap<Eigen::Tensor<T, 1, L>> x(a, m);
    switch (fn)
    {
    case 1:
        x = x.sqrt();
        break;
    case 2:
        x = x.abs();
        break;
    case 3:
        x = x.sigmoid();
        break;
    case 4:
        x = x.exp();
        break;
    case 5:
        x = x.log();
        break;
    case 6:
        x = x.sign();
        break;
    default:
        break;
    }
}

extern "C" void pg_tensor_unaryop(int oid,int fn,int c1,void* a1)
{
    if (oid == 700)
        tensor_unaryop<float, Eigen::RowMajor>(fn, c1, (float*) a1);
    else if (oid == 701)
        tensor_unaryop<double, Eigen::RowMajor>(fn, c1, (double*) a1);
    else if (oid ==  21)
        tensor_unaryop<short, Eigen::RowMajor>(fn, c1, (short*) a1);
    else if (oid ==  23)
        tensor_unaryop<int, Eigen::RowMajor>(fn, c1, (int*) a1);
    else if (oid ==  20)
        tensor_unaryop<long, Eigen::RowMajor>(fn, c1, (long*) a1);
}

template<typename T,int L,int M>
void tensor_convolve(T* i1,int* d1,T* k2,int* d2,int* s3,int* p4,T* o5,int* d5)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> in(i1, m);
    Eigen::array<int, M> n;
    for (int i=0;i < M;i++) n[i] = d2[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> kr(k2, n);
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, M-1> pd;
    if (p4 == NULL)
    {
        for (int i=0;i < M-1;i++)
            pd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    }
    else
    {
        for (int i=0;i < M-1;i++)
            pd[i] = std::make_pair((ptrdiff_t)p4[2*i+2],(ptrdiff_t)p4[2*i+3]);
    }
    Eigen::array<ptrdiff_t, M-1> cd;
    for (int i=0;i < M-1;i++) cd[i] = i;
    Eigen::array<ptrdiff_t, M-1> st;
    if (s3 == NULL)
        for (int i=0;i < M-1;i++) st[i] = 1;
    else
        for (int i=0;i < M-1;i++) st[i] = s3[i];
    Eigen::array<int, M-2> x;
    for (int i=0;i < M-2;i++) x[i] = d5[i+1];
    Eigen::array<int, M> z;
    z[0] = d5[0];z[1] = d5[M-1];
    for (int i=2;i < M;i++) z[i] = d5[i-1];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> ot(o5, z);
    for (int i = 0; i < in.dimension(0); i++)
    {
        for (int j = 0; j < kr.dimension(M-1); j++)
        {
            Eigen::Tensor<T, M-1, L> o1 = in.template chip<0>(i).pad(pd).convolve(kr.template chip<M-1>(j), cd).stride(st);
            Eigen::TensorMap<Eigen::Tensor<T, M-2, L>> o2(o1.data(), x);
            (ot.template chip<0>(i)).template chip<0>(j) = o2;
        }
    }
    Eigen::DSizes<Eigen::DenseIndex, M> si;
    si[0] = 0;si[M-1] = 1;
    for (int i=1;i < M-1;i++) si[i] = i+1;
    Eigen::Tensor<T, M, L> fo = ot.shuffle(si);
    std::copy(fo.data(), fo.data() + fo.size(), o5);
}

extern "C" void pg_tensor_convolve(int oid,void* i1,int n1,int* d1,void* k2,int* d2,int* s3,int* p4,void* o5,int* d5)
{
    if (oid == 700)
    {
        if (n1 == 3)
            tensor_convolve<float, Eigen::RowMajor, 3>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) o5, d5);
        else if (n1 == 4)
            tensor_convolve<float, Eigen::RowMajor, 4>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) o5, d5);
        else if (n1 == 5)
            tensor_convolve<float, Eigen::RowMajor, 5>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) o5, d5);
    }
    else if (oid == 701)
    {
        if (n1 == 3)
            tensor_convolve<double, Eigen::RowMajor, 3>((double*) i1, d1, (double*) k2, d2, s3, p4, (double*) o5, d5);
        else if (n1 == 4)
            tensor_convolve<double, Eigen::RowMajor, 4>((double*) i1, d1, (double*) k2, d2, s3, p4, (double*) o5, d5);
        else if (n1 == 5)
            tensor_convolve<double, Eigen::RowMajor, 5>((double*) i1, d1, (double*) k2, d2, s3, p4, (double*) o5, d5);
    }
}

template<typename T,int L,int M>
void tensor_pool(int fn,T* i1,int* d1,int* k2,int* s3,int* p4,T* o5,int* d5)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> in(i1, m);
    Eigen::array<int, M> kr;
    kr[0] = d1[0];kr[M-1] = d1[M-1];
    for (int i=1;i < M-1;i++) kr[i] = k2[i];
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, M> pd;
    if (p4 == NULL)
    {
        for (int i=0;i < M;i++)
            pd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    }
    else
    {
        for (int i=0;i < M;i++)
            pd[i] = std::make_pair((ptrdiff_t)p4[2*i],(ptrdiff_t)p4[2*i+1]);
    }
    Eigen::Tensor<T, 3, L> pp;
    Eigen::array<ptrdiff_t, M-2> rd;
    for (int i=2;i < M;i++) rd[i-2] = i;
    Eigen::Tensor<T, M, L> p0 = in.pad(pd);
    Eigen::Tensor<T, M+1, L> ep = p0.extract_patches(kr);
    if (fn == 1)
        pp = ep.maximum(rd);
    else if (fn == 2)
        pp = ep.mean(rd);
    Eigen::DSizes<Eigen::DenseIndex, M> sd;
    sd[M-2] = d1[0];sd[M-1] = d1[M-1];
    for (int i=0;i < M-2;i++) sd[i] = p0.dimension(i+1) - k2[i+1] + 1;
    Eigen::Tensor<T, M, L> ps = pp.reshape(sd);
    Eigen::DSizes<Eigen::DenseIndex, M> fd;
    fd[0] = M-2;fd[M-1] = M-1;
    for (int i=1;i < M-1;i++) fd[i] = i-1;
    Eigen::Tensor<T, M, L> pf = ps.shuffle(fd);
    Eigen::DSizes<Eigen::DenseIndex, M> st;
    for (int i=0;i < M;i++) st[i] = 0;
    Eigen::DSizes<Eigen::DenseIndex, M> ed;
    for (int i=0;i < M;i++) ed[i] = pf.dimension(i);
    Eigen::DSizes<Eigen::DenseIndex, M> iv;
    for (int i=0;i < M;i++) iv[i] = s3[i];
    Eigen::Tensor<T, M, L> fo = pf.stridedSlice(st,ed,iv);
    std::copy(fo.data(), fo.data() + fo.size(), o5);
}

extern "C" void pg_tensor_pool(int oid,int fn,void* i1,int n1,int* d1,int* k2,int* s3,int* p4,void* o5,int* d5)
{
    if (oid == 700)
    {
        if (n1 == 3)
            tensor_pool<float, Eigen::RowMajor, 3>(fn, (float*) i1, d1, k2, s3, p4, (float*) o5, d5);
        else if (n1 == 4)
            tensor_pool<float, Eigen::RowMajor, 4>(fn, (float*) i1, d1, k2, s3, p4, (float*) o5, d5);
        else if (n1 == 5)
            tensor_pool<float, Eigen::RowMajor, 5>(fn, (float*) i1, d1, k2, s3, p4, (float*) o5, d5);
    }
    else if (oid == 701)
    {
        if (n1 == 3)
            tensor_pool<double, Eigen::RowMajor, 3>(fn, (double*) i1, d1, k2, s3, p4, (double*) o5, d5);
        else if (n1 == 4)
            tensor_pool<double, Eigen::RowMajor, 4>(fn, (double*) i1, d1, k2, s3, p4, (double*) o5, d5);
        else if (n1 == 5)
            tensor_pool<double, Eigen::RowMajor, 5>(fn, (double*) i1, d1, k2, s3, p4, (double*) o5, d5);
    }
}

template<typename T,int L>
void tensor_activate(int fn,int m,T* a,T g)
{
    Eigen::TensorMap<Eigen::Tensor<T, 1, L>> x(a, m);
    switch (fn)
    {
        case 1:
            x = x.cwiseMax((T)0);
            break;
        case 2:
            x = x.sigmoid();
            break;
        case 3:
            x = x.tanh();
            break;
        case 4:
            x = x.cwiseMin((T)0) * g + x.cwiseMax((T)0);
            break;
        case 5:
            x = (x.cwiseMin((T)0).exp() - x.constant((T)1)) * g + x.cwiseMax((T)0);
            break;
        default:
            break;
    }
}

extern "C" void pg_tensor_activate(int oid,int fn,int c1,void* a1,float g)
{
    if (oid == 700)
        tensor_activate<float, Eigen::RowMajor>(fn, c1, (float*) a1, g);
    else if (oid == 701)
        tensor_activate<double, Eigen::RowMajor>(fn, c1, (double*) a1, (double) g);
}

template<typename T,int L,int M>
void tensor_dropout(T* i1,int* d1,T r2,int* n2,int s2)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> in(i1, m);
    Eigen::array<int, M> lo, hi, bc;
    for (int i=0;i < M;i++) lo[i] = 0;
    if (n2 == NULL)
    {
        for (int i=0;i < M;i++)
        {
            hi[i] = d1[i];
            bc[i] = 1;
        }
    }
    else
    {
        for (int i=0;i < M;i++)
        {
            hi[i] = n2[i];
            bc[i] = d1[i] / n2[i];
        }
    }
    Eigen::Tensor<T, M, L> ru = in.slice(lo, hi);
    Eigen::internal::UniformRandomGenerator<T> u(s2);
    Eigen::Tensor<bool, M, L> kp = ru.random(u) >= ru.constant(r2);
    Eigen::Tensor<T, M, L> km = kp.template cast<T>();
    in = in / in.constant(1 - r2) * km.broadcast(bc);
}

extern "C" void pg_tensor_dropout(int oid,void* i1,int n1,int* d1,float r2,int* n2,int s2)
{
    if (oid == 700)
    {
        if (n1 == 1)
            tensor_dropout<float, Eigen::RowMajor, 1>((float*) i1, d1, r2, n2, s2);
        else if (n1 == 2)
            tensor_dropout<float, Eigen::RowMajor, 2>((float*) i1, d1, r2, n2, s2);
        else if (n1 == 3)
            tensor_dropout<float, Eigen::RowMajor, 3>((float*) i1, d1, r2, n2, s2);
        else if (n1 == 4)
            tensor_dropout<float, Eigen::RowMajor, 4>((float*) i1, d1, r2, n2, s2);
        else if (n1 == 5)
            tensor_dropout<float, Eigen::RowMajor, 5>((float*) i1, d1, r2, n2, s2);
        else if (n1 == 6)
            tensor_dropout<float, Eigen::RowMajor, 6>((float*) i1, d1, r2, n2, s2);
    }
    else if (oid == 701)
    {
        if (n1 == 1)
            tensor_dropout<double, Eigen::RowMajor, 1>((double*) i1, d1, (double) r2, n2, s2);
        else if (n1 == 2)
            tensor_dropout<double, Eigen::RowMajor, 2>((double*) i1, d1, (double) r2, n2, s2);
        else if (n1 == 3)
            tensor_dropout<double, Eigen::RowMajor, 3>((double*) i1, d1, (double) r2, n2, s2);
        else if (n1 == 4)
            tensor_dropout<double, Eigen::RowMajor, 4>((double*) i1, d1, (double) r2, n2, s2);
        else if (n1 == 5)
            tensor_dropout<double, Eigen::RowMajor, 5>((double*) i1, d1, (double) r2, n2, s2);
        else if (n1 == 6)
            tensor_dropout<double, Eigen::RowMajor, 6>((double*) i1, d1, (double) r2, n2, s2);
    }
}

template<typename T,int L>
void tensor_rmatmul(int n1,T* i1,int* d1,T* i2,int* d2,bool* b2,T* o3,int* d3)
{
    int x1 = 1;
    for (int i=0;i < n1-2;i++) x1 *= d1[i];
    Eigen::array<int, 3> m = {{x1, d1[n1-2], d1[n1-1]}};
    Eigen::TensorMap<Eigen::Tensor<T, 3, L>> a(i1, m);
    Eigen::array<int, 3> n = {{x1, d2[n1-2], d2[n1-1]}};
    Eigen::TensorMap<Eigen::Tensor<T, 3, L>> b(i2, n);
    Eigen::Tensor<T, 3, L> lt = a, rt = b;
    if (b2 != NULL)
    {
        if (b2[0]) lt = a.shuffle(Eigen::DSizes<Eigen::DenseIndex, 3> (0,2,1));
        if (b2[1]) rt = b.shuffle(Eigen::DSizes<Eigen::DenseIndex, 3> (0,2,1));
    }
    Eigen::array<Eigen::Tensor<float, 1>::DimensionPair, 1> d = {Eigen::Tensor<float, 1>::DimensionPair(1, 0)};
    Eigen::array<int, 3> o = {{x1, d3[n1-2], d3[n1-1]}};
    Eigen::TensorMap<Eigen::Tensor<T, 3, L>> ot(o3, o);
    for (int i = 0; i < ot.dimension(0); ++i)
        ot.template chip<0>(i) = lt.template chip<0>(i).contract(rt.template chip<0>(i),d);
}

template<typename T,int L>
void tensor_cmatmul(int n1,std::complex<T>* i1,int* d1,std::complex<T>* i2,int* d2,bool* b2,std::complex<T>* o3,int* d3)
{
    int x1 = 1;
    for (int i=0;i < n1-3;i++) x1 *= d1[i];
    Eigen::array<int, 3> m = {{x1, d1[n1-3], d1[n1-2]}};
    Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 3, L>> a(i1, m);
    Eigen::array<int, 3> n = {{x1, d2[n1-3], d2[n1-2]}};
    Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 3, L>> b(i2, n);
    Eigen::Tensor<std::complex<T>, 3, L> lt = a, rt = b;
    if (b2 != NULL)
    {
        if (b2[2])
            lt = a.conjugate().shuffle(Eigen::DSizes<Eigen::DenseIndex, 3> (0,2,1));
        else if (b2[0])
            lt = a.shuffle(Eigen::DSizes<Eigen::DenseIndex, 3> (0,2,1));
        if (b2[3])
            rt = b.conjugate().shuffle(Eigen::DSizes<Eigen::DenseIndex, 3> (0,2,1));
        else if (b2[1])
            rt = b.shuffle(Eigen::DSizes<Eigen::DenseIndex, 3> (0,2,1));
    }
    Eigen::array<Eigen::Tensor<float, 1>::DimensionPair, 1> d = {Eigen::Tensor<float, 1>::DimensionPair(1, 0)};
    Eigen::array<int, 3> o = {{x1, d3[n1-3], d3[n1-2]}};
    Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 3, L>> ot(o3, o);
    for (int i = 0; i < ot.dimension(0); ++i)
        ot.template chip<0>(i) = lt.template chip<0>(i).contract(rt.template chip<0>(i),d);
}

extern "C" void pg_tensor_matmul(int oid,int m1,int n1,void* i1,int* d1,void* i2,int* d2,bool* b2,void* o3,int* d3)
{
    if (oid == 700)
    {
        if (n1 - m1 == 2)
            tensor_rmatmul<float, Eigen::RowMajor>(n1,(float*) i1, d1,(float*) i2, d2, b2,(float*) o3, d3);
        else if (n1 - m1 == 3)
            tensor_cmatmul<float, Eigen::RowMajor>(n1,(std::complex<float> *) i1, d1,(std::complex<float> *) i2, d2, b2,(std::complex<float> *) o3, d3);
    }
    else if (oid == 701)
    {
        if (n1 - m1 == 2)
            tensor_rmatmul<double, Eigen::RowMajor>(n1,(double*) i1, d1,(double*) i2, d2, b2,(double*) o3, d3);
        else if (n1 - m1 == 3)
            tensor_cmatmul<double, Eigen::RowMajor>(n1,(std::complex<double> *) i1, d1,(std::complex<double> *) i2, d2, b2,(std::complex<double> *) o3, d3);
    }
    else if (oid ==  21)
    {
        if (n1 - m1 == 2)
            tensor_rmatmul<short, Eigen::RowMajor>(n1,(short*) i1, d1,(short*) i2, d2, b2,(short*) o3, d3);
        else if (n1 - m1 == 3)
            tensor_cmatmul<short, Eigen::RowMajor>(n1,(std::complex<short> *) i1, d1,(std::complex<short> *) i2, d2, b2,(std::complex<short> *) o3, d3);
    }
    else if (oid ==  23)
    {
        if (n1 - m1 == 2)
            tensor_rmatmul<int, Eigen::RowMajor>(n1,(int*) i1, d1,(int*) i2, d2, b2,(int*) o3, d3);
        else if (n1 - m1 == 3)
            tensor_cmatmul<int, Eigen::RowMajor>(n1,(std::complex<int> *) i1, d1,(std::complex<int> *) i2, d2, b2,(std::complex<int> *) o3, d3);
    }
    else if (oid ==  20)
    {
        if (n1 - m1 == 2)
            tensor_rmatmul<long, Eigen::RowMajor>(n1,(long*) i1, d1,(long*) i2, d2, b2,(long*) o3, d3);
        else if (n1 - m1 == 3)
            tensor_cmatmul<long, Eigen::RowMajor>(n1,(std::complex<long> *) i1, d1,(std::complex<long> *) i2, d2, b2,(std::complex<long> *) o3, d3);
    }
}

template<typename T,int L,int M>
void tensor_softmax(T* in,int* d1,int ax,double* out)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(in, m);
    Eigen::TensorMap<Eigen::Tensor<double, M, L>> y(out, m);
    Eigen::Tensor<double, M, L> a = x.template cast<double>().exp();
    Eigen::array<Eigen::ptrdiff_t, 1> r = {ax};
    Eigen::array<Eigen::ptrdiff_t, M> s, t;
    for (int i=0;i < M;i++)
    {
        if (i == ax)
        { s[i] = 1; t[i] = d1[i]; }
        else
        { t[i] = 1; s[i] = d1[i]; }
    }
    y = a / (a.sum(r).reshape(s).broadcast(t));
}

extern "C" void pg_tensor_softmax(int oid,void* in,int n1,int* d1,int ax,void* out)
{
    if (oid == 700)
    {
        if (n1 == 1)
            tensor_softmax<float, Eigen::RowMajor, 1>((float*) in, d1, ax, (double*) out);
        else if (n1 == 2)
            tensor_softmax<float, Eigen::RowMajor, 2>((float*) in, d1, ax, (double*) out);
        else if (n1 == 3)
            tensor_softmax<float, Eigen::RowMajor, 3>((float*) in, d1, ax, (double*) out);
        else if (n1 == 4)
            tensor_softmax<float, Eigen::RowMajor, 4>((float*) in, d1, ax, (double*) out);
        else if (n1 == 5)
            tensor_softmax<float, Eigen::RowMajor, 5>((float*) in, d1, ax, (double*) out);
        else if (n1 == 6)
            tensor_softmax<float, Eigen::RowMajor, 6>((float*) in, d1, ax, (double*) out);
    }
    else if (oid == 701)
    {
        if (n1 == 1)
            tensor_softmax<double, Eigen::RowMajor, 1>((double*) in, d1, ax, (double*) out);
        else if (n1 == 2)
            tensor_softmax<double, Eigen::RowMajor, 2>((double*) in, d1, ax, (double*) out);
        else if (n1 == 3)
            tensor_softmax<double, Eigen::RowMajor, 3>((double*) in, d1, ax, (double*) out);
        else if (n1 == 4)
            tensor_softmax<double, Eigen::RowMajor, 4>((double*) in, d1, ax, (double*) out);
        else if (n1 == 5)
            tensor_softmax<double, Eigen::RowMajor, 5>((double*) in, d1, ax, (double*) out);
        else if (n1 == 6)
            tensor_softmax<double, Eigen::RowMajor, 6>((double*) in, d1, ax, (double*) out);
    }
    else if (oid ==  21)
    {
        if (n1 == 1)
            tensor_softmax<short, Eigen::RowMajor, 1>((short*) in, d1, ax, (double*) out);
        else if (n1 == 2)
            tensor_softmax<short, Eigen::RowMajor, 2>((short*) in, d1, ax, (double*) out);
        else if (n1 == 3)
            tensor_softmax<short, Eigen::RowMajor, 3>((short*) in, d1, ax, (double*) out);
        else if (n1 == 4)
            tensor_softmax<short, Eigen::RowMajor, 4>((short*) in, d1, ax, (double*) out);
        else if (n1 == 5)
            tensor_softmax<short, Eigen::RowMajor, 5>((short*) in, d1, ax, (double*) out);
        else if (n1 == 6)
            tensor_softmax<short, Eigen::RowMajor, 6>((short*) in, d1, ax, (double*) out);
    }
    else if (oid ==  23)
    {
        if (n1 == 1)
            tensor_softmax<int, Eigen::RowMajor, 1>((int*) in, d1, ax, (double*) out);
        else if (n1 == 2)
            tensor_softmax<int, Eigen::RowMajor, 2>((int*) in, d1, ax, (double*) out);
        else if (n1 == 3)
            tensor_softmax<int, Eigen::RowMajor, 3>((int*) in, d1, ax, (double*) out);
        else if (n1 == 4)
            tensor_softmax<int, Eigen::RowMajor, 4>((int*) in, d1, ax, (double*) out);
        else if (n1 == 5)
            tensor_softmax<int, Eigen::RowMajor, 5>((int*) in, d1, ax, (double*) out);
        else if (n1 == 6)
            tensor_softmax<int, Eigen::RowMajor, 6>((int*) in, d1, ax, (double*) out);
    }
    else if (oid ==  20)
    {
        if (n1 == 1)
            tensor_softmax<long, Eigen::RowMajor, 1>((long*) in, d1, ax, (double*) out);
        else if (n1 == 2)
            tensor_softmax<long, Eigen::RowMajor, 2>((long*) in, d1, ax, (double*) out);
        else if (n1 == 3)
            tensor_softmax<long, Eigen::RowMajor, 3>((long*) in, d1, ax, (double*) out);
        else if (n1 == 4)
            tensor_softmax<long, Eigen::RowMajor, 4>((long*) in, d1, ax, (double*) out);
        else if (n1 == 5)
            tensor_softmax<long, Eigen::RowMajor, 5>((long*) in, d1, ax, (double*) out);
        else if (n1 == 6)
            tensor_softmax<long, Eigen::RowMajor, 6>((long*) in, d1, ax, (double*) out);
    }
}

template<typename T,int L,int M>
void tensor_argpos(int fn,T* in,int* d1,int ax,long* out)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(in, m);
    Eigen::Tensor<Eigen::DenseIndex, M-1, L> y;
    if (fn == 1)
        y = x.argmax(ax);
    else if (fn == 2)
        y = x.argmin(ax);
    std::copy(y.data(), y.data() + y.size(), out);
}

extern "C" void pg_tensor_argpos(int oid,int fn,char* in,int n1,int* d1,void* out,int ax)
{
    if (oid == 700)
    {
        if (n1 == 1)
            tensor_argpos<float, Eigen::RowMajor, 1>(fn, (float*) in, d1, ax, (long*) out);
        else if (n1 == 2)
            tensor_argpos<float, Eigen::RowMajor, 2>(fn, (float*) in, d1, ax, (long*) out);
        else if (n1 == 3)
            tensor_argpos<float, Eigen::RowMajor, 3>(fn, (float*) in, d1, ax, (long*) out);
        else if (n1 == 4)
            tensor_argpos<float, Eigen::RowMajor, 4>(fn, (float*) in, d1, ax, (long*) out);
        else if (n1 == 5)
            tensor_argpos<float, Eigen::RowMajor, 5>(fn, (float*) in, d1, ax, (long*) out);
        else if (n1 == 6)
            tensor_argpos<float, Eigen::RowMajor, 6>(fn, (float*) in, d1, ax, (long*) out);
    }
    else if (oid == 701)
    {
        if (n1 == 1)
            tensor_argpos<double, Eigen::RowMajor, 1>(fn, (double*) in, d1, ax, (long*) out);
        else if (n1 == 2)
            tensor_argpos<double, Eigen::RowMajor, 2>(fn, (double*) in, d1, ax, (long*) out);
        else if (n1 == 3)
            tensor_argpos<double, Eigen::RowMajor, 3>(fn, (double*) in, d1, ax, (long*) out);
        else if (n1 == 4)
            tensor_argpos<double, Eigen::RowMajor, 4>(fn, (double*) in, d1, ax, (long*) out);
        else if (n1 == 5)
            tensor_argpos<double, Eigen::RowMajor, 5>(fn, (double*) in, d1, ax, (long*) out);
        else if (n1 == 6)
            tensor_argpos<double, Eigen::RowMajor, 6>(fn, (double*) in, d1, ax, (long*) out);
    }
    else if (oid ==  21)
    {
        if (n1 == 1)
            tensor_argpos<short, Eigen::RowMajor, 1>(fn, (short*) in, d1, ax, (long*) out);
        else if (n1 == 2)
            tensor_argpos<short, Eigen::RowMajor, 2>(fn, (short*) in, d1, ax, (long*) out);
        else if (n1 == 3)
            tensor_argpos<short, Eigen::RowMajor, 3>(fn, (short*) in, d1, ax, (long*) out);
        else if (n1 == 4)
            tensor_argpos<short, Eigen::RowMajor, 4>(fn, (short*) in, d1, ax, (long*) out);
        else if (n1 == 5)
            tensor_argpos<short, Eigen::RowMajor, 5>(fn, (short*) in, d1, ax, (long*) out);
        else if (n1 == 6)
            tensor_argpos<short, Eigen::RowMajor, 6>(fn, (short*) in, d1, ax, (long*) out);
    }
    else if (oid ==  23)
    {
        if (n1 == 1)
            tensor_argpos<int, Eigen::RowMajor, 1>(fn, (int*) in, d1, ax, (long*) out);
        else if (n1 == 2)
            tensor_argpos<int, Eigen::RowMajor, 2>(fn, (int*) in, d1, ax, (long*) out);
        else if (n1 == 3)
            tensor_argpos<int, Eigen::RowMajor, 3>(fn, (int*) in, d1, ax, (long*) out);
        else if (n1 == 4)
            tensor_argpos<int, Eigen::RowMajor, 4>(fn, (int*) in, d1, ax, (long*) out);
        else if (n1 == 5)
            tensor_argpos<int, Eigen::RowMajor, 5>(fn, (int*) in, d1, ax, (long*) out);
        else if (n1 == 6)
            tensor_argpos<int, Eigen::RowMajor, 6>(fn, (int*) in, d1, ax, (long*) out);
    }
    else if (oid ==  20)
    {
        if (n1 == 1)
            tensor_argpos<long, Eigen::RowMajor, 1>(fn, (long*) in, d1, ax, (long*) out);
        else if (n1 == 2)
            tensor_argpos<long, Eigen::RowMajor, 2>(fn, (long*) in, d1, ax, (long*) out);
        else if (n1 == 3)
            tensor_argpos<long, Eigen::RowMajor, 3>(fn, (long*) in, d1, ax, (long*) out);
        else if (n1 == 4)
            tensor_argpos<long, Eigen::RowMajor, 4>(fn, (long*) in, d1, ax, (long*) out);
        else if (n1 == 5)
            tensor_argpos<long, Eigen::RowMajor, 5>(fn, (long*) in, d1, ax, (long*) out);
        else if (n1 == 6)
            tensor_argpos<long, Eigen::RowMajor, 6>(fn, (long*) in, d1, ax, (long*) out);
    }
}

template<typename T,int L,int M>
void tensor_mean_absolute_error(T* i1,int* d1,T* i2,int ax,double* o3)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::array<Eigen::ptrdiff_t, 1> r = {ax};
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(i1, m);
    Eigen::Tensor<double, M, L> a = x.template cast<double>();
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> y(i2, m);
    Eigen::Tensor<double, M, L> b = y.template cast<double>();
    Eigen::Tensor<double, M-1, L> c = (b - a).abs().mean(r);
    std::copy(c.data(), c.data() + c.size(), o3);
}

template<typename T,int L,int M>
void tensor_mean_squared_error(T* i1,int* d1,T* i2,int ax,double* o3)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::array<Eigen::ptrdiff_t, 1> r = {ax};
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(i1, m);
    Eigen::Tensor<double, M, L> a = x.template cast<double>();
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> y(i2, m);
    Eigen::Tensor<double, M, L> b = y.template cast<double>();
    Eigen::Tensor<double, M-1, L> c = (b - a).pow(2).mean(r);
    std::copy(c.data(), c.data() + c.size(), o3);
}

template<typename T,int L,int M>
void tensor_categorical_cross_entropy(T* i1,int* d1,T* i2,int ax,double* o3)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::array<Eigen::ptrdiff_t, 1> r = {ax};
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(i1, m);
    Eigen::Tensor<double, M, L> a = x.template cast<double>().log();
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> y(i2, m);
    Eigen::Tensor<double, M, L> b = y.template cast<double>();
    Eigen::Tensor<double, M-1, L> c = - (a * b).sum(r);
    std::copy(c.data(), c.data() + c.size(), o3);
}

template<typename T,int L,int M>
void tensor_softmax_cross_entropy(T* i1,int* d1,T* i2,int ax,double* o3)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(i1, m);
    Eigen::Tensor<double, M, L> a = x.template cast<double>().exp();
    Eigen::array<Eigen::ptrdiff_t, 1> r = {ax};
    Eigen::array<Eigen::ptrdiff_t, M> s, t;
    for (int i=0;i < M;i++)
    {
        if (i == ax)
        { s[i] = 1; t[i] = d1[i]; }
        else
        { t[i] = 1; s[i] = d1[i]; }
    }
    Eigen::Tensor<double, M, L> d = a.sum(r).reshape(s).broadcast(t);
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> y(i2, m);
    Eigen::Tensor<double, M, L> b = y.template cast<double>();
    Eigen::Tensor<double, M-1, L> c = - (b * (a / d).log()).sum(r);
    std::copy(c.data(), c.data() + c.size(), o3);
}

extern "C" void pg_tensor_loss(int oid,int fn,void* i1,int n1,int* d1,void* i2,void* o3,int ax)
{
    if (oid == 700)
    {
        if (n1 == 1)
        {
            if (fn == 1)
                tensor_mean_absolute_error<float, Eigen::RowMajor, 1>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<float, Eigen::RowMajor, 1>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<float, Eigen::RowMajor, 1>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<float, Eigen::RowMajor, 1>((float*) i1, d1, (float*) i2, ax, (double*) o3);
        }
        else if (n1 == 2)
        {
            if (fn == 1)
                tensor_mean_absolute_error<float, Eigen::RowMajor, 2>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<float, Eigen::RowMajor, 2>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<float, Eigen::RowMajor, 2>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<float, Eigen::RowMajor, 2>((float*) i1, d1, (float*) i2, ax, (double*) o3);
        }
        else if (n1 == 3)
        {
            if (fn == 1)
                tensor_mean_absolute_error<float, Eigen::RowMajor, 3>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<float, Eigen::RowMajor, 3>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<float, Eigen::RowMajor, 3>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<float, Eigen::RowMajor, 3>((float*) i1, d1, (float*) i2, ax, (double*) o3);
        }
        else if (n1 == 4)
        {
            if (fn == 1)
                tensor_mean_absolute_error<float, Eigen::RowMajor, 4>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<float, Eigen::RowMajor, 4>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<float, Eigen::RowMajor, 4>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<float, Eigen::RowMajor, 4>((float*) i1, d1, (float*) i2, ax, (double*) o3);
        }
        else if (n1 == 5)
        {
            if (fn == 1)
                tensor_mean_absolute_error<float, Eigen::RowMajor, 5>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<float, Eigen::RowMajor, 5>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<float, Eigen::RowMajor, 5>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<float, Eigen::RowMajor, 5>((float*) i1, d1, (float*) i2, ax, (double*) o3);
        }
        else if (n1 == 6)
        {
            if (fn == 1)
                tensor_mean_absolute_error<float, Eigen::RowMajor, 6>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<float, Eigen::RowMajor, 6>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<float, Eigen::RowMajor, 6>((float*) i1, d1, (float*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<float, Eigen::RowMajor, 6>((float*) i1, d1, (float*) i2, ax, (double*) o3);
        }
    }
    else if (oid == 701)
    {
        if (n1 == 1)
        {
            if (fn == 1)
                tensor_mean_absolute_error<double, Eigen::RowMajor, 1>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<double, Eigen::RowMajor, 1>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<double, Eigen::RowMajor, 1>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<double, Eigen::RowMajor, 1>((double*) i1, d1, (double*) i2, ax, (double*) o3);
        }
        else if (n1 == 2)
        {
            if (fn == 1)
                tensor_mean_absolute_error<double, Eigen::RowMajor, 2>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<double, Eigen::RowMajor, 2>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<double, Eigen::RowMajor, 2>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<double, Eigen::RowMajor, 2>((double*) i1, d1, (double*) i2, ax, (double*) o3);
        }
        else if (n1 == 3)
        {
            if (fn == 1)
                tensor_mean_absolute_error<double, Eigen::RowMajor, 3>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<double, Eigen::RowMajor, 3>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<double, Eigen::RowMajor, 3>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<double, Eigen::RowMajor, 3>((double*) i1, d1, (double*) i2, ax, (double*) o3);
        }
        else if (n1 == 4)
        {
            if (fn == 1)
                tensor_mean_absolute_error<double, Eigen::RowMajor, 4>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<double, Eigen::RowMajor, 4>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<double, Eigen::RowMajor, 4>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<double, Eigen::RowMajor, 4>((double*) i1, d1, (double*) i2, ax, (double*) o3);
        }
        else if (n1 == 5)
        {
            if (fn == 1)
                tensor_mean_absolute_error<double, Eigen::RowMajor, 5>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<double, Eigen::RowMajor, 5>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<double, Eigen::RowMajor, 5>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<double, Eigen::RowMajor, 5>((double*) i1, d1, (double*) i2, ax, (double*) o3);
        }
        else if (n1 == 6)
        {
            if (fn == 1)
                tensor_mean_absolute_error<double, Eigen::RowMajor, 6>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 2)
                tensor_mean_squared_error<double, Eigen::RowMajor, 6>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 3)
                tensor_categorical_cross_entropy<double, Eigen::RowMajor, 6>((double*) i1, d1, (double*) i2, ax, (double*) o3);
            else if (fn == 4)
                tensor_softmax_cross_entropy<double, Eigen::RowMajor, 6>((double*) i1, d1, (double*) i2, ax, (double*) o3);
        }
    }
}

template<typename T,int L,int M>
void tensor_unpool(int fn,T* i1,int* d1,int* k2,int* s3,int* p4,T* g5,int* d5,T* o6)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> in(i1, m);
    Eigen::array<int, M> n;
    for (int i=0;i < M;i++) n[i] = d5[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gi(g5, n);
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, M> pd;
    if (p4 == NULL)
        for (int i=0;i < M;i++) pd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    else
        for (int i=0;i < M;i++) pd[i] = std::make_pair((ptrdiff_t)p4[2*i],(ptrdiff_t)p4[2*i+1]);
    Eigen::array<ptrdiff_t, M> bs;
    for (int i=0;i < M;i++) bs[i] = s3[i];
    Eigen::Tensor<T, M, L> gb = gi.broadcast(bs);
    Eigen::array<Eigen::ptrdiff_t, 2*M-2> md;
    md[0] = gb.dimension(0); md[2*M-3] = gb.dimension(M-1);
    for (int i=1;i < 2*M-3;i=i+2) {md[i] = s3[(i+1)/2]; md[i+1] = d5[(i+2)/2];}
    Eigen::Tensor<T, 2*M-2, L> gr = gb.reshape(md);
    Eigen::DSizes<Eigen::DenseIndex, 2*M-2> sx;
    sx[0] = 0;sx[2*M-3] = 2*M-3;
    for (int i=1;i < 2*M-3;i=i+2) {sx[i] = i+1; sx[i+1] = i;}
    Eigen::Tensor<T, 2*M-2, L> gf = gr.shuffle(sx);
    Eigen::array<Eigen::ptrdiff_t, M> sd;
    for (int i=0;i < M;i++) sd[i] = d5[i] * s3[i];
    Eigen::Tensor<T, M, L> gt = gf.reshape(sd);
    if (fn == 1)
    {
        Eigen::array<int, M> kr;
        kr[0] = d1[0];kr[M-1] = d1[M-1];
        for (int i=1;i < M-1;i++) kr[i] = k2[i];
        Eigen::array<ptrdiff_t, M-2> rd;
        for (int i=2;i < M;i++) rd[i-2] = i;
        Eigen::Tensor<T, M, L> p0 = in.pad(pd);
        Eigen::Tensor<T, M+1, L> ep = p0.extract_patches(kr);
        Eigen::Tensor<Eigen::Tuple<Eigen::DenseIndex, T>, M+1, L> et = ep.index_tuples();
        Eigen::Tensor<Eigen::Tuple<Eigen::DenseIndex, T>, 3, L> em = et.reduce(rd, Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<Eigen::DenseIndex, T>>());
        Eigen::Tensor<T, M+1, L> vp = ep.constant(0);
        for (int i = 0; i < em.size(); ++i) vp(em(i).first) = 1;
        Eigen::array<Eigen::ptrdiff_t, 2*M-2> ds;
        for (int i=0;i < M-2;i++) ds[i] = in.dimension(i+1);
        for (int i=M-2;i < 2*M-2;i++) ds[i] = kr[i-(M-2)];
        Eigen::Tensor<T, 2*M-2, L> rp = vp.reshape(ds);
        Eigen::DSizes<Eigen::DenseIndex, 2*M-2> st;
        for (int i=0;i < 2*M-2;i++) st[i] = 0;
        Eigen::DSizes<Eigen::DenseIndex, 2*M-2> ed;
        for (int i=0;i < 2*M-2;i++) ed[i] = rp.dimension(i);
        Eigen::DSizes<Eigen::DenseIndex, 2*M-2> iv;
        for (int i=0;i < M-2;i++) iv[i] = s3[i+1];
        for (int i=M-2;i < 2*M-2;i++) iv[i] = 1;
        Eigen::Tensor<T, 2*M-2, L> sp = rp.stridedSlice(st,ed,iv);
        Eigen::DSizes<Eigen::DenseIndex, 2*M-2> si;
        si[0] = M-2; si[2*M-3] = 2*M-3;
        for (int i=1;i < 2*M-3;i=i+2) {si[i] = i/2; si[i+1] = (M-1) + (i/2);}
        Eigen::Tensor<T, 2*M-2, L> fp = sp.shuffle(si);
        gt *= fp.reshape(sd);
    }
    else if (fn == 2)
    {
        Eigen::TensorMap<Eigen::Tensor<int, 1, L>> ke(k2, M);
        Eigen::Tensor<T, 0, L> kp = ke.prod().template cast<T>();
        gt /= gt.constant(kp());
    }
    Eigen::array<ptrdiff_t, M> s;
    for (int i=0;i < M;i++) s[i] = pd[i].first;
    Eigen::Tensor<T, M, L> ft = gt.slice(s, m);
    std::copy(ft.data(), ft.data() + ft.size(), o6);
}

extern "C" void pg_tensor_unpool(int oid,int fn,void* i1,int n1,int* d1,int* k2,int* s3,int* p4,void* g5,int* d5,void* o6)
{
    if (oid == 700)
    {
        if (n1 == 3)
            tensor_unpool<float, Eigen::RowMajor, 3>(fn, (float*) i1, d1, k2, s3, p4, (float*) g5, d5, (float*) o6);
        else if (n1 == 4)
            tensor_unpool<float, Eigen::RowMajor, 4>(fn, (float*) i1, d1, k2, s3, p4, (float*) g5, d5, (float*) o6);
        else if (n1 == 5)
            tensor_unpool<float, Eigen::RowMajor, 5>(fn, (float*) i1, d1, k2, s3, p4, (float*) g5, d5, (float*) o6);
    }
    else if (oid == 701)
    {
        if (n1 == 3)
            tensor_unpool<double, Eigen::RowMajor, 3>(fn, (double*) i1, d1, k2, s3, p4, (double*) g5, d5, (double*) o6);
        else if (n1 == 4)
            tensor_unpool<double, Eigen::RowMajor, 4>(fn, (double*) i1, d1, k2, s3, p4, (double*) g5, d5, (double*) o6);
        else if (n1 == 5)
            tensor_unpool<double, Eigen::RowMajor, 5>(fn, (double*) i1, d1, k2, s3, p4, (double*) g5, d5, (double*) o6);
    }
}

template<typename T,int L,int M>
void tensor_convt(T* i1,int* d1,T* k2,int* d2,int* s3,int* p4,T* g5,int* d5,T* o6,T* o7,T* o8)
{
    Eigen::array<int, M> m;
    for (int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> in(i1, m);
    Eigen::array<int, M> n;
    for (int i=0;i < M;i++) n[i] = d2[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> kr(k2, n);
    Eigen::array<int, M> q;
    for (int i=0;i < M;i++) q[i] = d5[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gi(g5, q);
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, M> pd, bd, xd;
    if (p4 == NULL)
        for (int i=0;i < M;i++) pd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    else
        for (int i=0;i < M;i++) pd[i] = std::make_pair((ptrdiff_t)p4[2*i],(ptrdiff_t)p4[2*i+1]);
    Eigen::Tensor<T, M, L> p0 = in.pad(pd);
    if (s3 == NULL)
        for (int i=0;i < M;i++) bd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    else
        for (int i=0;i < M;i++) bd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)(d5[i] * (s3[i] - 1)));
    Eigen::Tensor<T, M, L> gy = gi.pad(bd);
    if (gy.size() != gi.size())
    {
        Eigen::array<Eigen::ptrdiff_t, 2*M-2> md;
        md[0] = gy.dimension(0);md[2*M-3] = gy.dimension(M-1);
        for (int i=1;i < 2*M-3;i=i+2) {md[i] = s3[(i+1)/2];md[i+1] = d5[(i+2)/2];}
        Eigen::Tensor<T, 2*M-2, L> gr = gy.reshape(md);
        Eigen::DSizes<Eigen::DenseIndex, 2*M-2> sx;
        sx[0] = 0;sx[2*M-3] = 2*M-3;
        for (int i=1;i < 2*M-3;i=i+2) {sx[i] = i+1; sx[i+1] = i;}
        Eigen::Tensor<T, 2*M-2, L> gf = gr.shuffle(sx);
        Eigen::array<Eigen::ptrdiff_t, M> sd;
        for (int i=0;i < M;i++) sd[i] = d5[i] * s3[i];
        Eigen::Tensor<T, M, L> gp = gf.reshape(sd);
        Eigen::array<ptrdiff_t, M> st, ed;
        for (int i=0;i < M;i++) st[i] = 0;
        ed[0] = gp.dimension(0);ed[M-1] = gp.dimension(M-1);
        for (int i=1;i < M-1;i++) ed[i] = p0.dimension(i) - n[i-1] + 1;
        gy = gp.slice(st, ed);
    }
    Eigen::array<ptrdiff_t, M-1> cd;
    for (int i=0;i < M-1;i++) cd[i] = i;
    Eigen::array<int, M-2> zd;
    for (int i=0;i < M-2;i++) zd[i] = n[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gw(o6, n);
    for (int i = 0; i < p0.dimension(M-1); i++)
    {
        for (int j = 0; j < gy.dimension(M-1); j++)
        {
            Eigen::Tensor<T, M-1, L> cv = p0.template chip<M-1>(i).convolve(gy.template chip<M-1>(j), cd);
            (gw.template chip<M-1>(j)).template chip<M-2>(i) = cv.reshape(zd);
        }
    }
    for (int i=0;i < M-2;i++) zd[i] = m[i+1];
    xd[0] = xd[M-1] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    for (int i=1;i < M-1;i++) xd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)(m[i] + n[i-1] - 1 - gy.dimension(i)));
    Eigen::Tensor<T, M, L> gy_ = gy.pad(xd);
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gx(o7, m);
    for (int i = 0; i < gy_.dimension(0); i++)
    {
        for (int j = 0; j < kr.dimension(M-2); j++)
        {
            Eigen::Tensor<T, M-1, L> cv = gy_.template chip<0>(i).convolve(kr.template chip<M-2>(j), cd);
            (gx.template chip<M-1>(j)).template chip<0>(i) = cv.reshape(zd);
        }
    }
    Eigen::array<int, M-1> ud, kd;
    for (int i=0;i < M-1;i++) {ud[i] = 1;kd[i] = q[i];}
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> gb(o8, q);
    for (int i = 0; i < gb.dimension(M-1); i++)
    {
        Eigen::Tensor<T, M-1, L> sb = gi.template chip<M-1>(i).sum().reshape(ud);
        gb.template chip<M-1>(i) = sb.broadcast(kd);
    }
}

extern "C" void pg_tensor_convt(int oid,void* i1,int n1,int* d1,void* k2,int* d2,int* s3,int* p4,void* g5,int* d5,void* o6,void* o7,void* o8)
{
    if (oid == 700)
    {
        if (n1 == 3)
            tensor_convt<float, Eigen::RowMajor, 3>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) g5, d5, (float*) o6, (float*) o7, (float*) o8);
        else if (n1 == 4)
            tensor_convt<float, Eigen::RowMajor, 4>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) g5, d5, (float*) o6, (float*) o7, (float*) o8);
        else if (n1 == 5)
            tensor_convt<float, Eigen::RowMajor, 5>((float*) i1, d1, (float*) k2, d2, s3, p4, (float*) g5, d5, (float*) o6, (float*) o7, (float*) o8);
    }
    else if (oid == 701)
    {
        if (n1 == 3)
            tensor_convt<double, Eigen::RowMajor, 3>((double*) i1, d1, (double*) k2, d2, s3, p4, (double*) g5, d5, (double*) o6, (double*) o7, (double*) o8);
        else if (n1 == 4)
            tensor_convt<double, Eigen::RowMajor, 4>((double*) i1, d1, (double*) k2, d2, s3, p4, (double*) g5, d5, (double*) o6, (double*) o7, (double*) o8);
        else if (n1 == 5)
            tensor_convt<double, Eigen::RowMajor, 5>((double*) i1, d1, (double*) k2, d2, s3, p4, (double*) g5, d5, (double*) o6, (double*) o7, (double*) o8);
    }
}
