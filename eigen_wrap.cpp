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
