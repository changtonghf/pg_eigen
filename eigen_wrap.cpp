#include <iostream>
#include <array>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename T,int L,unsigned int M>
void tensor_reduce(unsigned int fn,T* in,unsigned int* d1,T* out)
{
    Eigen::array<unsigned int, M> m;
    for (unsigned int i=0;i < M;i++) m[i] = d1[i];
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

template<typename T,int L,unsigned int M,unsigned int N>
void tensor_reduce(unsigned int fn,T* in,unsigned int* d1,T* out,unsigned int* d2)
{
    Eigen::array<unsigned int, M> m;
    for (unsigned int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(in, m);
    Eigen::array<unsigned int, N> r;
    for (unsigned int i=0;i < N;i++) r[i] = d2[i];
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

extern "C" void pg_tensor_reduce(unsigned int oid,unsigned int fn,char* in,unsigned int n1,unsigned int* d1,void* out,unsigned int n2,unsigned int* d2)
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

template<typename T,int L,unsigned int M,unsigned int R,unsigned int D>
void tensor_rfft(T* in,unsigned int* d1,T* out,unsigned int* d2)
{
    Eigen::array<unsigned int, M> m;
    for (unsigned int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> x(in, m);
    Eigen::array<unsigned int, M> f;
    for (unsigned int i=0;i < M;i++) f[i] = d2[i];
    Eigen::Tensor<std::complex<T>, M, L> y = x.template fft<R, D>(f);
    for (Eigen::DenseIndex i=0;i < y.size();i++)
    {
        out[2*i] = y(i).real();
        out[2*i+1] = y(i).imag();
    }
}

template<typename T,int L,unsigned int M,unsigned int R,unsigned int D>
void tensor_fft(std::complex<T>* in,unsigned int* d1,T* out,unsigned int* d2)
{
    Eigen::array<unsigned int, M> m;
    for (unsigned int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<std::complex<T>, M, L>> x(in, m);
    Eigen::array<unsigned int, M> f;
    for (unsigned int i=0;i < M;i++) f[i] = d2[i];
    Eigen::Tensor<std::complex<T>, M, L> y = x.template fft<R, D>(f);
    for (Eigen::DenseIndex i=0;i < y.size();i++)
    {
        out[2*i] = y(i).real();
        out[2*i+1] = y(i).imag();
    }
}

extern "C" void pg_tensor_fft(unsigned int oid,bool forward,char* in,unsigned int n1,unsigned int* d1,void* out,unsigned int n2,unsigned int* d2)
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

extern "C" void pg_tensor_random(unsigned int fn,unsigned int num,double* out,double a1,double b1)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    if (fn == 1)
    {
        std::normal_distribution<double> nd(a1, b1);
        for (unsigned int i = 0; i < num; ++i)
            out[i] = nd(gen);
    }
    else if (fn == 2)
    {
        double lb = a1 - 2 * b1, ub = a1 + 2 * b1;
        std::normal_distribution<double> td(a1, b1);
        for (unsigned int i = 0; i < num; ++i)
        {
            double tmp;
            do { tmp = td(gen); } while (tmp < lb || tmp > ub);
            out[i] = tmp;
        }
    }
    else if (fn == 3)
    {
        std::uniform_real_distribution<double> ud(a1, b1);
        for (unsigned int i = 0; i < num; ++i)
            out[i] = ud(gen);
    }
    else if (fn == 4)
    {
        std::gamma_distribution<double> gd(a1, b1);
        for (unsigned int i = 0; i < num; ++i)
            out[i] = gd(gen);
    }
}

extern "C" void pg_tensor_shuffle(unsigned int oid,unsigned int step, unsigned int num,void* out)
{
    std::default_random_engine r{std::random_device{}()};
    if (oid == 700)
    {
        for (unsigned int i = 0;(i * step) < num;i++)
        {
            std::shuffle((float*)out, (float*)out + step, r);
            out = (float*)out + step;
        }
    }
    else if (oid == 701)
    {
        for (unsigned int i = 0;(i * step) < num;i++)
        {
            std::shuffle((double*)out, (double*)out + step, r);
            out = (double*)out + step;
        }
    }
    else if (oid ==  21)
    {
        for (unsigned int i = 0;(i * step) < num;i++)
        {
            std::shuffle((short*)out, (short*)out + step, r);
            out = (short*)out + step;
        }
    }
    else if (oid ==  23)
    {
        for (unsigned int i = 0;(i * step) < num;i++)
        {
            std::shuffle((int*)out, (int*)out + step, r);
            out = (int*)out + step;
        }
    }
    else if (oid ==  20)
    {
        for (unsigned int i = 0;(i * step) < num;i++)
        {
            std::shuffle((long*)out, (long*)out + step, r);
            out = (long*)out + step;
        }
    }
}

template<typename T,int L>
void tensor_unaryop(unsigned int fn,unsigned int m,T* a,T* b)
{
    Eigen::TensorMap<Eigen::Tensor<T, 1, L>> x(a, m);
    Eigen::TensorMap<Eigen::Tensor<T, 1, L>> y(b, m);
    if (fn == 1)
        x = x + y;
    else if (fn == 2)
        x = x - y;
    else if (fn == 3)
        x = x * y;
    else if (fn == 4)
        x = x / y;
}

extern "C" void pg_tensor_calc(unsigned int oid,unsigned int fn,unsigned int num,void* a1,void* a2)
{
    if (oid == 700)
        tensor_unaryop<float, Eigen::RowMajor>(fn, num, (float*) a1, (float*) a2);
    else if (oid == 701)
        tensor_unaryop<double, Eigen::RowMajor>(fn, num, (double*) a1, (double*) a2);
    else if (oid ==  21)
        tensor_unaryop<short, Eigen::RowMajor>(fn, num, (short*) a1, (short*) a2);
    else if (oid ==  23)
        tensor_unaryop<int, Eigen::RowMajor>(fn, num, (int*) a1, (int*) a2);
    else if (oid ==  20)
        tensor_unaryop<long, Eigen::RowMajor>(fn, num, (long*) a1, (long*) a2);
}
