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

template<typename T,int L,unsigned int M>
void tensor_convolve(T* i1,unsigned int* d1,T* k2,unsigned int* d2,unsigned int* s3,unsigned int* p4,T* o5,unsigned int* d5)
{
    Eigen::array<unsigned int, M> m;
    for (unsigned int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> in(i1, m);
    Eigen::array<unsigned int, M> n;
    for (unsigned int i=0;i < M;i++) n[i] = d2[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> kr(k2, n);
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, M-1> pd;
    if (p4 == NULL)
    {
        for (unsigned int i=0;i < M-1;i++)
            pd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    }
    else
    {
        for (unsigned int i=0;i < M-1;i++)
            pd[i] = std::make_pair((ptrdiff_t)p4[2*i],(ptrdiff_t)p4[2*i+1]);
    }
    Eigen::array<ptrdiff_t, M-1> cd;
    for (unsigned int i=0;i < M-1;i++) cd[i] = i;
    Eigen::array<ptrdiff_t, M-1> st;
    if (s3 == NULL)
        for (unsigned int i=0;i < M-1;i++) st[i] = 1;
    else
        for (unsigned int i=0;i < M-1;i++) st[i] = s3[i];
    Eigen::array<unsigned int, M-2> x;
    for (unsigned int i=0;i < M-2;i++) x[i] = d5[i+1];
    Eigen::array<unsigned int, M> z;
    z[0] = d5[0];z[1] = d5[M-1];
    for (unsigned int i=2;i < M;i++) z[i] = d5[i-1];
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
    for (unsigned int i=1;i < M-1;i++) si[i] = i+1;
    Eigen::Tensor<T, M, L> fo = ot.shuffle(si);
    std::copy(fo.data(), fo.data() + fo.size(), o5);
}

extern "C" void pg_tensor_convolve(unsigned int oid,void* i1,unsigned int n1,unsigned int* d1,void* k2,unsigned int* d2,unsigned int* s3,unsigned int* p4,void* o5,unsigned int* d5)
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

template<typename T,int L,unsigned int M>
void tensor_pool(unsigned int fn,T* i1,unsigned int* d1,unsigned int* k2,unsigned int* s3,unsigned int* p4,T* o5,unsigned int* d5)
{
    Eigen::array<unsigned int, M> m;
    for (unsigned int i=0;i < M;i++) m[i] = d1[i];
    Eigen::TensorMap<Eigen::Tensor<T, M, L>> in(i1, m);
    Eigen::array<unsigned int, M> kr;
    kr[0] = d1[0];kr[M-1] = d1[M-1];
    for (unsigned int i=1;i < M-1;i++) kr[i] = k2[i];
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, M> pd;
    if (p4 == NULL)
    {
        for (unsigned int i=0;i < M;i++)
            pd[i] = std::make_pair((ptrdiff_t)0,(ptrdiff_t)0);
    }
    else
    {
        for (unsigned int i=0;i < M;i++)
            pd[i] = std::make_pair((ptrdiff_t)p4[2*i],(ptrdiff_t)p4[2*i+1]);
    }
    Eigen::Tensor<T, 3, L> pp;
    Eigen::array<ptrdiff_t, M-2> rd;
    for (unsigned int i=2;i < M;i++) rd[i-2] = i;
    Eigen::Tensor<T, M, L> p0 = in.pad(pd);
    Eigen::Tensor<T, M+1, L> ep = p0.extract_patches(kr);
    if (fn == 1)
        pp = ep.maximum(rd);
    else if (fn == 2)
        pp = ep.mean(rd);
    Eigen::DSizes<Eigen::DenseIndex, M> sd;
    sd[M-2] = d1[0];sd[M-1] = d1[M-1];
    for (unsigned int i=0;i < M-2;i++) sd[i] = p0.dimension(i+1) - k2[i+1] + 1;
    Eigen::Tensor<T, M, L> ps = pp.reshape(sd);
    Eigen::DSizes<Eigen::DenseIndex, M> fd;
    fd[0] = M-2;fd[M-1] = M-1;
    for (unsigned int i=1;i < M-1;i++) fd[i] = i-1;
    Eigen::Tensor<T, M, L> pf = ps.shuffle(fd);
    Eigen::DSizes<Eigen::DenseIndex, M> st;
    for (unsigned int i=0;i < M;i++) st[i] = 0;
    Eigen::DSizes<Eigen::DenseIndex, M> ed;
    for (unsigned int i=0;i < M;i++) ed[i] = pf.dimension(i);
    Eigen::DSizes<Eigen::DenseIndex, M> iv;
    for (unsigned int i=0;i < M;i++) iv[i] = s3[i];
    Eigen::Tensor<T, M, L> fo = pf.stridedSlice(st,ed,iv);
    std::copy(fo.data(), fo.data() + fo.size(), o5);
}

extern "C" void pg_tensor_pool(unsigned int oid,unsigned int fn,void* i1,unsigned int n1,unsigned int* d1,unsigned int* k2,unsigned int* s3,unsigned int* p4,void* o5,unsigned int* d5)
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
