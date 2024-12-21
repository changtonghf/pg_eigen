#include "postgres.h"
#include "fmgr.h"
#include "utils/elog.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/arrayaccess.h"
#include "catalog/pg_type.h"
#include "executor/instrument.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

extern void pg_tensor_reduce(unsigned int oid,unsigned int fn,char* in,unsigned int n1,unsigned int* d1,void* out,unsigned int n2,unsigned int* d2);
extern void pg_tensor_fft(unsigned int oid,bool forward,char* in,unsigned int n1,unsigned int* d1,void* out,unsigned int n2,unsigned int* d2);
extern void pg_tensor_random(unsigned int fn,unsigned int num,double* out,double a1,double b1);
extern void pg_tensor_shuffle(unsigned int oid,unsigned int step,unsigned int num,void* out);
extern void pg_tensor_calc(unsigned int oid,unsigned int fn,unsigned int num,void* a1,void* a2);
extern void pg_tensor_convolve(unsigned int oid,void* i1,unsigned int n1,unsigned int* d1,void* k2,unsigned int* d2,unsigned int* s3,unsigned int* p4,void* o5,unsigned int* d5);

PG_FUNCTION_INFO_V1(array_reduce);
PG_FUNCTION_INFO_V1(array_fft);
PG_FUNCTION_INFO_V1(array_random);
PG_FUNCTION_INFO_V1(array_shuffle);
PG_FUNCTION_INFO_V1(array_calc);
PG_FUNCTION_INFO_V1(array_convolve);

Datum array_reduce(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3;
    char      *fn, *v1;
    Oid        t1,  t2;
    int        n1,  n2,  c2,  l3,  m3 = 0, n3 = 1;
    int       *d1 = NULL, *d2 = NULL, *d3, *b3, x[6] = {0,0,0,0,0,0};
    uint32    *p2 = NULL;
    instr_time s1,  s2;
    void      *v2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "reducer function name not specified.");
    fn = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    if (strcasecmp(fn, "sum") != 0 && strcasecmp(fn, "mean") != 0 && strcasecmp(fn, "prod") != 0 && strcasecmp(fn, "maximum") != 0 && strcasecmp(fn, "minimum") != 0)
        elog(ERROR, "\"%s\" is currently not supported in tensor reduce.", fn);
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != INT2OID && t1 != INT4OID && t1 != INT8OID && t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "the second array argument type must be number type.");
    if (ARR_HASNULL(a1))
        elog(ERROR, "the second array elements can't be null.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    v1 = ARR_DATA_PTR(a1);

    if (PG_ARGISNULL(2))
    {
        m3 = 1;
        x[0] = 1;
    }
    else
    {
        a2 = PG_GETARG_ARRAYTYPE_P(2);
        if (ARR_HASNULL(a2))
            elog(ERROR, "the third array elements can't be null.");
        t2 = ARR_ELEMTYPE(a2);
        if (t2 != INT4OID)
            elog(ERROR, "the third array argument type must be integer type.");
        n2 = ARR_NDIM(a2);
        if (n2 > 1)
            elog(ERROR, "the third argument must be one-dimensional.");
        d2 = ARR_DIMS(a2);
        c2 = ArrayGetNItems(n2, d2);
        if (c2 > n1)
            elog(ERROR, "the third array argument length must be less than or equal to the second array argument dimension.");
        p2 = (uint32 *)ARR_DATA_PTR(a2);
        for (uint32 i=0;i < c2;i++)
        {
            if (p2[i] > n1 && p2[i] < 0)
                elog(ERROR, "the value of third array element is out of range.");
            for (uint32 j=i+1;j < c2;j++)
            {
                if (p2[i] == p2[j])
                    elog(ERROR, "the value of third array element must be unique.");
            }
        }
        for (uint32 i=0;i < n1;i++)
        {
            bool flag = false;
            for (uint32 j=0;j < c2;j++)
            {
                if (i == p2[j])
                {
                    flag = true;
                    break;
                }
            }
            if (!flag)
            {
                x[m3] = d1[i];
                n3 *= d1[i];
                m3++;
            }
        }
        if (m3 == 0)
        {
            m3 = 1;
            x[0] = 1;
        }
    }

    if (t1 == FLOAT4OID)
    {
        v2 = palloc(n3 * sizeof(float4));
        l3 = n3 * sizeof(float4) + ARR_OVERHEAD_NONULLS(m3);
    }
    else if (t1 == FLOAT8OID)
    {
        v2 = palloc(n3 * sizeof(float8));
        l3 = n3 * sizeof(float8) + ARR_OVERHEAD_NONULLS(m3);
    }
    else if (t1 == INT2OID)
    {
        v2 = palloc(n3 * sizeof(int16));
        l3 = n3 * sizeof(int16) + ARR_OVERHEAD_NONULLS(m3);
    }
    else if (t1 == INT4OID)
    {
        v2 = palloc(n3 * sizeof(int32));
        l3 = n3 * sizeof(int32) + ARR_OVERHEAD_NONULLS(m3);
    }
    else if (t1 == INT8OID)
    {
        v2 = palloc(n3 * sizeof(int64));
        l3 = n3 * sizeof(int64) + ARR_OVERHEAD_NONULLS(m3);
    }
    d3 = (int *) palloc(m3 * sizeof(int));
    b3 = (int *) palloc(m3 * sizeof(int));
    for (uint32 i=0;i < m3;i++)
    {
        d3[i] = x[i];
        b3[i] = 1;
    }

    INSTR_TIME_SET_CURRENT(s1);

    if (strcasecmp(fn, "sum") == 0)
        pg_tensor_reduce(t1, 1, v1, n1, (unsigned int*)d1, v2, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "mean") == 0)
        pg_tensor_reduce(t1, 2, v1, n1, (unsigned int*)d1, v2, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "prod") == 0)
        pg_tensor_reduce(t1, 3, v1, n1, (unsigned int*)d1, v2, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "maximum") == 0)
        pg_tensor_reduce(t1, 4, v1, n1, (unsigned int*)d1, v2, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "minimum") == 0)
        pg_tensor_reduce(t1, 5, v1, n1, (unsigned int*)d1, v2, (d2 ? d2[0] : 0), p2);

    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen reduce spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

    a3 = (ArrayType *) palloc0(l3);
    SET_VARSIZE(a3, l3);
    a3->ndim = m3;
    a3->dataoffset = 0;
    a3->elemtype = t1;
    memcpy(ARR_DIMS(a3)  , d3, m3 * sizeof(int));
    memcpy(ARR_LBOUND(a3), b3, m3 * sizeof(int));
    if (t1 == FLOAT4OID)
        memcpy(ARR_DATA_PTR(a3), v2, n3 * sizeof(float4));
    else if (t1 == FLOAT8OID)
        memcpy(ARR_DATA_PTR(a3), v2, n3 * sizeof(float8));
    else if (t1 == INT2OID)
        memcpy(ARR_DATA_PTR(a3), v2, n3 * sizeof(int16 ));
    else if (t1 == INT4OID)
        memcpy(ARR_DATA_PTR(a3), v2, n3 * sizeof(int32 ));
    else if (t1 == INT8OID)
        memcpy(ARR_DATA_PTR(a3), v2, n3 * sizeof(int64 ));
    pfree(v2);
    pfree(d3);
    pfree(b3);
    PG_RETURN_ARRAYTYPE_P(a3);
}

Datum array_fft(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3;
    char      *v1;
    bool       fw;
    Oid        t1,  t2;
    int        n1,  n2,  c2,  l3,  m3 = 0, n3 = 1;
    int       *d1 = NULL, *d2 = NULL, *d3, *b3, x[6] = {0,0,0,0,0,0};
    uint32    *p2 = NULL;
    instr_time s1,  s2;
    void      *v2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "discrete fourier transform direction not specified.");
    fw = PG_GETARG_BOOL(0);
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    if (PG_ARGISNULL(2))
        elog(ERROR, "discrete fourier transform axes not specified.");
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "the second array argument type must be number type.");
    if (ARR_HASNULL(a1))
        elog(ERROR, "the second array elements can't be null.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    v1 = ARR_DATA_PTR(a1);

    a2 = PG_GETARG_ARRAYTYPE_P(2);
    if (ARR_HASNULL(a2))
        elog(ERROR, "the third array elements can't be null.");
    t2 = ARR_ELEMTYPE(a2);
    if (t2 != INT4OID)
        elog(ERROR, "the third array argument type must be integer type.");
    n2 = ARR_NDIM(a2);
    if (n2 > 1)
        elog(ERROR, "the third argument must be one-dimensional.");
    d2 = ARR_DIMS(a2);
    c2 = ArrayGetNItems(n2, d2);
    if (n1 != c2 && n1 != c2 + 1)
        elog(ERROR, "the length of the third argument should be either equal to or one less than the dimension of the second argument.");
    if (n1 == c2 + 1 && d1[n1-1] != 2)
        elog(ERROR, "the second argument's last dimension should only have two elements representing a complex.");
    p2 = (uint32 *)ARR_DATA_PTR(a2);
    for (uint32 i=0;i < c2;i++)
    {
        if (p2[i] > n1 && p2[i] < 0)
            elog(ERROR, "the value of third array element is out of range.");
        for (uint32 j=i+1;j < c2;j++)
        {
            if (p2[i] == p2[j])
                elog(ERROR, "the value of third array element must be unique.");
        }
    }
    for (uint32 i=0;i < n1;i++)
    {
        x[m3] = d1[i];
        n3 *= d1[i];
        m3++;
    }
    if (n1 == c2)
    {
        x[n1] = 2;
        n3 *= 2;
        m3++;
    }

    if (t1 == FLOAT4OID)
    {
        v2 = palloc(n3 * sizeof(float4));
        l3 = n3 * sizeof(float4) + ARR_OVERHEAD_NONULLS(m3);
    }
    else if (t1 == FLOAT8OID)
    {
        v2 = palloc(n3 * sizeof(float8));
        l3 = n3 * sizeof(float8) + ARR_OVERHEAD_NONULLS(m3);
    }
    d3 = (int *) palloc(m3 * sizeof(int));
    b3 = (int *) palloc(m3 * sizeof(int));
    for (uint32 i=0;i < m3;i++)
    {
        d3[i] = x[i];
        b3[i] = 1;
    }

    INSTR_TIME_SET_CURRENT(s1);
    pg_tensor_fft(t1, fw, v1, n1, (unsigned int*)d1, v2, c2, p2);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen discrete fourier transform spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

    a3 = (ArrayType *) palloc0(l3);
    SET_VARSIZE(a3, l3);
    a3->ndim = m3;
    a3->dataoffset = 0;
    a3->elemtype = t1;
    memcpy(ARR_DIMS(a3)  , d3, m3 * sizeof(int));
    memcpy(ARR_LBOUND(a3), b3, m3 * sizeof(int));
    if (t1 == FLOAT4OID)
        memcpy(ARR_DATA_PTR(a3), v2, n3 * sizeof(float4));
    else if (t1 == FLOAT8OID)
        memcpy(ARR_DATA_PTR(a3), v2, n3 * sizeof(float8));
    pfree(v2);
    pfree(d3);
    pfree(b3);
    PG_RETURN_ARRAYTYPE_P(a3);
}

Datum array_random(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2;
    char      *fn;
    uint32    *p1;
    int        n1, *d1, c1, l2, *b2, *d2, n2 = 1;
    float8     a, b, *v2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "random distribution type not specified.");
    if (PG_ARGISNULL(1))
        elog(ERROR, "tensor dimension array not specified.");
    fn = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (strcasecmp(fn, "random_normal") != 0 && strcasecmp(fn, "truncated_normal") != 0 && strcasecmp(fn, "random_uniform") != 0 && strcasecmp(fn, "random_gamma") != 0)
        elog(ERROR, "\"%s\" is currently not supported in tensor random.", fn);
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    c1 = ArrayGetNItems(n1, d1);
    p1 = (uint32 *)ARR_DATA_PTR(a1);
    if (PG_ARGISNULL(2))
    {
        if (strcasecmp(fn, "random_gamma") == 0)
            a = 1;
        else
            a = 0;
    }
    else
        a = PG_GETARG_FLOAT8(2);
    if (PG_ARGISNULL(3))
        b = 1;
    else
        b = PG_GETARG_FLOAT8(3);

    d2 = (int *) palloc(c1 * sizeof(int));
    b2 = (int *) palloc(c1 * sizeof(int));
    for (uint32 i=0;i < c1;i++) 
    {
        d2[i] = p1[i];
        b2[i] = 1;
        n2 *= p1[i];
    }
    v2 = (float8 *) palloc(n2 * sizeof(float8));

    if (strcasecmp(fn, "random_normal") == 0)
        pg_tensor_random(1, n2, v2, a, b);
    else if (strcasecmp(fn, "truncated_normal") == 0)
        pg_tensor_random(2, n2, v2, a, b);
    else if (strcasecmp(fn, "random_uniform") == 0)
        pg_tensor_random(3, n2, v2, a, b);
    else if (strcasecmp(fn, "random_gamma") == 0)
        pg_tensor_random(4, n2, v2, a, b);

    l2 = n2 * sizeof(float8) + ARR_OVERHEAD_NONULLS(c1);
    a2 = (ArrayType *) palloc0(l2);
    SET_VARSIZE(a2, l2);
    a2->ndim = c1;
    a2->dataoffset = 0;
    a2->elemtype = FLOAT8OID;
    memcpy(ARR_DIMS(a2)  , d2, c1 * sizeof(int));
    memcpy(ARR_LBOUND(a2), b2, c1 * sizeof(int));
    memcpy(ARR_DATA_PTR(a2), v2, n2 * sizeof(float8));
    pfree(v2);
    pfree(b2);
    pfree(d2);
    PG_RETURN_ARRAYTYPE_P(a2);
}

Datum array_shuffle(PG_FUNCTION_ARGS)
{
    ArrayType *a1;
    char      *p1;
    Oid        t1;
    int        n1, *d1, c1, d2, s2;

    if (PG_ARGISNULL(0))
        PG_RETURN_NULL();
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != INT2OID && t1 != INT4OID && t1 != INT8OID && t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "the second array argument type must be number type.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    c1 = ArrayGetNItems(n1, d1);
    p1 = ARR_DATA_PTR(a1);
    if (PG_ARGISNULL(1))
        d2 = n1;
    else
        d2 = PG_GETARG_INT32(1);
    if (d2 > n1)
        elog(ERROR, "the second argument should be either equal to or one less than the dimension of the first argument.");
    s2 = c1;
    if (d2 != n1)
    {
        for (uint32 i=0;i < d2;i++) s2 /= d1[i];
    }
    pg_tensor_shuffle(t1, s2, c1, (void*) p1);
    PG_RETURN_ARRAYTYPE_P(a1);
}

Datum array_calc(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2;
    char      *fn, *p1, *p2;
    Oid        t1,  t2;
    int        n1,  n2, c1, *d1, *d2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "calculate function name not specified.");
    fn = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (strcasecmp(fn, "add") != 0 && strcasecmp(fn, "sub") != 0 && strcasecmp(fn, "mul") != 0 && strcasecmp(fn, "div") != 0)
        elog(ERROR, "\"%s\" is currently not supported in tensor broadcasting calculation.", fn);
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    if (PG_ARGISNULL(2)) PG_RETURN_NULL();
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != INT2OID && t1 != INT4OID && t1 != INT8OID && t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "array argument type must be number type.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    c1 = ArrayGetNItems(n1, d1);
    a2 = PG_GETARG_ARRAYTYPE_P(2);
    t2 = ARR_ELEMTYPE(a2);
    if (t1 != t2)
        elog(ERROR, "array argument type must be same currently.");
    n2 = ARR_NDIM(a2);
    d2 = ARR_DIMS(a2);
    if (n1 != n2)
        elog(ERROR, "array argument dimension must be same.");
    for (uint32 i=0;i < n1;i++)
    {
        if (d1[i] != d2[i])
            elog(ERROR, "array argument dimension must be same.");
    }
    p1 = ARR_DATA_PTR(a1);
    p2 = ARR_DATA_PTR(a2);

    if (strcasecmp(fn, "add") == 0)
        pg_tensor_calc(t1, 1, c1, (void*) p1, (void*) p2);
    else if (strcasecmp(fn, "sub") == 0)
        pg_tensor_calc(t1, 2, c1, (void*) p1, (void*) p2);
    else if (strcasecmp(fn, "mul") == 0)
        pg_tensor_calc(t1, 3, c1, (void*) p1, (void*) p2);
    else if (strcasecmp(fn, "div") == 0)
        pg_tensor_calc(t1, 4, c1, (void*) p1, (void*) p2);

    PG_RETURN_ARRAYTYPE_P(a1);
}

Datum array_convolve(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3, *a5;
    char      *df, *pd, *p1, *p2, *p3;
    void      *p5;
    Oid        t1,  t2;
    int        n1, *d1, n2, *d2, n3, *d3, c3, c4, *p4;
    int        n5, *d5, *b5, c5 = 1, l5, s[6] = {0,0,0,0,0,0};
    instr_time s1,  s2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "data format not specified.");
    df = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (strcasecmp(df, "NWC") != 0 && strcasecmp(df, "NHWC") != 0 && strcasecmp(df, "NDHWC") != 0)
        elog(ERROR, "\"%s\" is not supported in tensor convolution input.", df);
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    if (PG_ARGISNULL(2))
        elog(ERROR, "convolution kernel not specified.");
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    a2 = PG_GETARG_ARRAYTYPE_P(2);
    if (PG_ARGISNULL(3))
    {
        a3 = NULL;
        n3 = 0;
        d3 = NULL;
        p3 = NULL;
    }
    else
    {
        a3 = PG_GETARG_ARRAYTYPE_P(3);
        n3 = ARR_NDIM(a3);
        if (n3 != 1) elog(ERROR, "strides shape must be one dimension.");
        d3 = ARR_DIMS(a3);
        c3 = ArrayGetNItems(n3, d3);
        p3 = ARR_DATA_PTR(a3);
    }
    if (PG_ARGISNULL(4))
        elog(ERROR, "padding type not specified.");
    else
    {
        pd = text_to_cstring(PG_GETARG_TEXT_P(4));
        if (strcasecmp(pd, "SAME") != 0 && strcasecmp(pd, "VALID") != 0)
            elog(ERROR, "\"%s\" is not supported in tensor convolution padding.", pd);
    }
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "input argument type must be float array type.");
    t2 = ARR_ELEMTYPE(a2);
    if (t2 != FLOAT4OID && t2 != FLOAT8OID)
        elog(ERROR, "kernel argument type must be float array type.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    n2 = ARR_NDIM(a2);
    d2 = ARR_DIMS(a2);
    p1 = ARR_DATA_PTR(a1);
    p2 = ARR_DATA_PTR(a2);
    s[0] = d1[0];
    for (uint32 i=0;i < n1-1;i++)
    {
        if (((int32*)p3)[i] <= 0 || ((int32*)p3)[i] > d1[i+1])
            elog(ERROR, "strides shape does not meet conv%dd operation.", n1-2);
    }
    if (strcasecmp(df, "NWC") == 0)
    {
        s[2] = d2[2];
        n5 = 3;
    }
    else if (strcasecmp(df, "NHWC") == 0)
    {
        s[3] = d2[3];
        n5 = 4;
    }
    else if (strcasecmp(df, "NDHWC") == 0)
    {
        s[4] = d2[4];
        n5 = 5;
    }
    if (n1 != n5 || n2 != n5)
        elog(ERROR, "input or kernel shape does not meet conv%dd operation.", n5-2);
    for (uint32 i=0;i < n5-1;i++)
        if (d2[i] > d1[i+1]) elog(ERROR, "input or kernel shape does not meet conv%dd operation.", n5-2);
    if (a3 != NULL && (c3 != n5-1 || ((int32*)p3)[c3-1] != 1))
        elog(ERROR, "strides shape does not meet conv%dd operation.", n5-2);
    if (strcasecmp(pd, "SAME") == 0)
    {
        if (n3 == 0)
        {
            for (uint32 i=1;i < n5-1;i++) s[i] = d1[i];
        }
        else
        {
            for (uint32 i=1;i < n5-1;i++)
            {
                if (d1[i] % (((int32*)p3)[i-1]) == 0)
                    s[i] = d1[i] / (((int32*)p3)[i-1]);
                else
                    s[i] = d1[i] / (((int32*)p3)[i-1]) + 1;
            }
        }
        c4 = (n5-1) * 2;
        p4 = (int *) palloc0(c4 * sizeof(int));
        for (uint32 i=0;i < n5-2;i++)
        {
            if (d2[i] > ((int32*)p3)[i])
            {
                p4[2*i] = (d2[i] - 1) / 2;
                p4[2*i+1] = ((d2[i] - 1) / 2) + ((d2[i] - 1) % 2);
            }
            else
            {
                int32 _p_ = s[i+1] * (((int32*)p3)[i]) - d1[i+1];
                p4[2*i] = _p_ / 2;
                p4[2*i+1] = (_p_ / 2) + (_p_ % 2);
            }
        }
    }
    else
    {
        if (n3 == 0)
        {
            for (uint32 i=1;i < n5-1;i++) s[i] = d1[i] - d2[i-1] + 1;
        }
        else
        {
            for (uint32 i=1;i < n5-1;i++)
            {
                if ((d1[i] - d2[i-1] + 1) % (((int32*)p3)[i-1]) == 0)
                    s[i] = (d1[i] - d2[i-1] + 1) / (((int32*)p3)[i-1]);
                else
                    s[i] = (d1[i] - d2[i-1] + 1) / (((int32*)p3)[i-1]) + 1;
            }
        }
        c4 = 0;
        p4 = NULL;
    }
    d5 = (int *) palloc(n5 * sizeof(int));
    b5 = (int *) palloc(n5 * sizeof(int));
    for (uint32 i=0;i < n5;i++)
    {
        d5[i] = s[i];
        b5[i] = 1;
        c5 *= s[i];
    }
    if (t1 == FLOAT4OID)
    {
        p5 = palloc(c5 * sizeof(float4));
        l5 = c5 * sizeof(float4) + ARR_OVERHEAD_NONULLS(n5);
    }
    else if (t1 == FLOAT8OID)
    {
        p5 = palloc(c5 * sizeof(float8));
        l5 = c5 * sizeof(float8) + ARR_OVERHEAD_NONULLS(n5);
    }

    INSTR_TIME_SET_CURRENT(s1);
    pg_tensor_convolve(t1, (void*) p1, n1, (unsigned int*)d1, (void*) p2, (unsigned int*)d2, (unsigned int*) p3, (unsigned int*) p4, (void*) p5, (unsigned int*)d5);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen convolution spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

    a5 = (ArrayType *) palloc0(l5);
    SET_VARSIZE(a5, l5);
    a5->ndim = n5;
    a5->dataoffset = 0;
    a5->elemtype = t1;
    memcpy(ARR_DIMS(a5)  , d5, n5 * sizeof(int));
    memcpy(ARR_LBOUND(a5), b5, n5 * sizeof(int));
    if (t1 == FLOAT4OID)
        memcpy(ARR_DATA_PTR(a5), p5, c5 * sizeof(float4));
    else if (t1 == FLOAT8OID)
        memcpy(ARR_DATA_PTR(a5), p5, c5 * sizeof(float8));
    pfree(p5);
    pfree(d5);
    pfree(b5);
    if (p4 != NULL) pfree(p4);
    PG_RETURN_ARRAYTYPE_P(a5);
}
