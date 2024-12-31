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

extern void pg_tensor_reduce(int oid,int fn,char* in,int n1,int* d1,void* out,int n2,int* d2);
extern void pg_tensor_fft(int oid,bool forward,char* in,int n1,int* d1,void* out,int n2,int* d2);
extern void pg_tensor_random(int fn,int c1,double* out,double a1,double b1,int s1);
extern void pg_tensor_shuffle(int oid,int s1,int c1,void* out);
extern void pg_tensor_binaryop(int oid,int fn,int c1,void* a1,void* a2);
extern void pg_tensor_convolve(int oid,void* i1,int n1,int* d1,void* k2,int* d2,int* s3,int* p4,void* o5,int* d5);
extern void pg_tensor_pool(int oid,int fn,void* i1,int n1,int* d1,int* k2,int* s3,int* p4,void* o5,int* d5);
extern void pg_tensor_activate(int oid,int fn,int c1,void* a1,float g);
extern void pg_tensor_dropout(int oid,void* i1,int n1,int* d1,float r2,int* n2,int s2);
extern void pg_tensor_matmul(int oid,int m1,int n1,void* i1,int* d1,void* i2,int* d2,bool* b2,void* o3,int* d3);
extern void pg_tensor_softmax(int oid,void* in,int n1,int* d1,int ax,void* out);
extern void pg_tensor_argpos(int oid,int fn,char* in,int n1,int* d1,void* out,int ax);
extern void pg_tensor_loss(int oid,int fn,void* i1,int n1,int* d1,void* i2,void* o3,int ax);

PG_FUNCTION_INFO_V1(array_reduce);
PG_FUNCTION_INFO_V1(array_fft);
PG_FUNCTION_INFO_V1(array_random);
PG_FUNCTION_INFO_V1(array_shuffle);
PG_FUNCTION_INFO_V1(array_binaryop);
PG_FUNCTION_INFO_V1(array_convolve);
PG_FUNCTION_INFO_V1(array_pool);
PG_FUNCTION_INFO_V1(array_activate);
PG_FUNCTION_INFO_V1(array_dropout);
PG_FUNCTION_INFO_V1(array_matmul);
PG_FUNCTION_INFO_V1(array_softmax);
PG_FUNCTION_INFO_V1(array_argpos);
PG_FUNCTION_INFO_V1(array_loss);

Datum array_reduce(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3;
    char      *fn, *p1;
    Oid        t1,  t2;
    int        n1,  n2,  c2,  l3,  n3 = 0, c3 = 1;
    int       *d1, *d2, *p2, *d3, *b3, x[6] = {0,0,0,0,0,0};
    void      *v3;
    instr_time s1,  s2;

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
    p1 = ARR_DATA_PTR(a1);

    if (PG_ARGISNULL(2))
    {
        n3 = 1;
        x[0] = 1;
        d2 = NULL;
        p2 = NULL;
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
        p2 = (int*)ARR_DATA_PTR(a2);
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
                x[n3] = d1[i];
                c3 *= d1[i];
                n3++;
            }
        }
        if (n3 == 0)
        {
            n3 = 1;
            x[0] = 1;
        }
    }

    if (t1 == FLOAT4OID)
    {
        v3 = palloc(c3 * sizeof(float4));
        l3 = c3 * sizeof(float4) + ARR_OVERHEAD_NONULLS(n3);
    }
    else if (t1 == FLOAT8OID)
    {
        v3 = palloc(c3 * sizeof(float8));
        l3 = c3 * sizeof(float8) + ARR_OVERHEAD_NONULLS(n3);
    }
    else if (t1 == INT2OID)
    {
        v3 = palloc(c3 * sizeof(int16));
        l3 = c3 * sizeof(int16) + ARR_OVERHEAD_NONULLS(n3);
    }
    else if (t1 == INT4OID)
    {
        v3 = palloc(c3 * sizeof(int32));
        l3 = c3 * sizeof(int32) + ARR_OVERHEAD_NONULLS(n3);
    }
    else if (t1 == INT8OID)
    {
        v3 = palloc(c3 * sizeof(int64));
        l3 = c3 * sizeof(int64) + ARR_OVERHEAD_NONULLS(n3);
    }
    d3 = (int *) palloc(n3 * sizeof(int));
    b3 = (int *) palloc(n3 * sizeof(int));
    for (uint32 i=0;i < n3;i++)
    {
        d3[i] = x[i];
        b3[i] = 1;
    }

    INSTR_TIME_SET_CURRENT(s1);

    if (strcasecmp(fn, "sum") == 0)
        pg_tensor_reduce(t1, 1, p1, n1, d1, v3, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "mean") == 0)
        pg_tensor_reduce(t1, 2, p1, n1, d1, v3, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "prod") == 0)
        pg_tensor_reduce(t1, 3, p1, n1, d1, v3, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "maximum") == 0)
        pg_tensor_reduce(t1, 4, p1, n1, d1, v3, (d2 ? d2[0] : 0), p2);
    else if (strcasecmp(fn, "minimum") == 0)
        pg_tensor_reduce(t1, 5, p1, n1, d1, v3, (d2 ? d2[0] : 0), p2);

    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen reduce spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

    a3 = (ArrayType *) palloc0(l3);
    SET_VARSIZE(a3, l3);
    a3->ndim = n3;
    a3->dataoffset = 0;
    a3->elemtype = t1;
    memcpy(ARR_DIMS(a3)  , d3, n3 * sizeof(int));
    memcpy(ARR_LBOUND(a3), b3, n3 * sizeof(int));
    if (t1 == FLOAT4OID)
        memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(float4));
    else if (t1 == FLOAT8OID)
        memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(float8));
    else if (t1 == INT2OID)
        memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(int16 ));
    else if (t1 == INT4OID)
        memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(int32 ));
    else if (t1 == INT8OID)
        memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(int64 ));
    pfree(v3);
    pfree(d3);
    pfree(b3);
    PG_RETURN_ARRAYTYPE_P(a3);
}

Datum array_fft(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3;
    char      *p1;
    bool       fw;
    Oid        t1,  t2;
    int        n1,  n2,  c2,  l3,  n3 = 0, c3 = 1;
    int       *d1, *d2, *p2, *d3, *b3, x[6] = {0,0,0,0,0,0};
    instr_time s1,  s2;
    void      *v3;

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
    p1 = ARR_DATA_PTR(a1);

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
    p2 = (int*)ARR_DATA_PTR(a2);
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
        x[n3] = d1[i];
        c3 *= d1[i];
        n3++;
    }
    if (n1 == c2)
    {
        x[n1] = 2;
        c3 *= 2;
        n3++;
    }

    if (t1 == FLOAT4OID)
    {
        v3 = palloc(c3 * sizeof(float4));
        l3 = c3 * sizeof(float4) + ARR_OVERHEAD_NONULLS(n3);
    }
    else if (t1 == FLOAT8OID)
    {
        v3 = palloc(c3 * sizeof(float8));
        l3 = c3 * sizeof(float8) + ARR_OVERHEAD_NONULLS(n3);
    }
    d3 = (int *) palloc(n3 * sizeof(int));
    b3 = (int *) palloc(n3 * sizeof(int));
    for (uint32 i=0;i < n3;i++)
    {
        d3[i] = x[i];
        b3[i] = 1;
    }

    INSTR_TIME_SET_CURRENT(s1);
    pg_tensor_fft(t1, fw, p1, n1, d1, v3, c2, p2);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen discrete fourier transform spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

    a3 = (ArrayType *) palloc0(l3);
    SET_VARSIZE(a3, l3);
    a3->ndim = n3;
    a3->dataoffset = 0;
    a3->elemtype = t1;
    memcpy(ARR_DIMS(a3)  , d3, n3 * sizeof(int));
    memcpy(ARR_LBOUND(a3), b3, n3 * sizeof(int));
    if (t1 == FLOAT4OID)
        memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(float4));
    else if (t1 == FLOAT8OID)
        memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(float8));
    pfree(v3);
    pfree(d3);
    pfree(b3);
    PG_RETURN_ARRAYTYPE_P(a3);
}

Datum array_random(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2;
    char      *fn;
    int        n1, *d1, *p1, c1;
    int        l2, *b2, *d2, c2 = 1, s;
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
    p1 = (int*)ARR_DATA_PTR(a1);
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
    if (PG_ARGISNULL(4))
        s = -1;
    else
    {
        s = PG_GETARG_INT32(4);
        if (s < 0) elog(ERROR, "random seed can't be less than 0.");
    }
    d2 = (int *) palloc(c1 * sizeof(int));
    b2 = (int *) palloc(c1 * sizeof(int));
    for (uint32 i=0;i < c1;i++) 
    {
        d2[i] = p1[i];
        b2[i] = 1;
        c2 *= p1[i];
    }
    v2 = (float8 *) palloc(c2 * sizeof(float8));

    if (strcasecmp(fn, "random_normal") == 0)
        pg_tensor_random(1, c2, v2, a, b, s);
    else if (strcasecmp(fn, "truncated_normal") == 0)
        pg_tensor_random(2, c2, v2, a, b, s);
    else if (strcasecmp(fn, "random_uniform") == 0)
        pg_tensor_random(3, c2, v2, a, b, s);
    else if (strcasecmp(fn, "random_gamma") == 0)
        pg_tensor_random(4, c2, v2, a, b, s);

    l2 = c2 * sizeof(float8) + ARR_OVERHEAD_NONULLS(c1);
    a2 = (ArrayType *) palloc0(l2);
    SET_VARSIZE(a2, l2);
    a2->ndim = c1;
    a2->dataoffset = 0;
    a2->elemtype = FLOAT8OID;
    memcpy(ARR_DIMS(a2)  , d2, c1 * sizeof(int));
    memcpy(ARR_LBOUND(a2), b2, c1 * sizeof(int));
    memcpy(ARR_DATA_PTR(a2), v2, c2 * sizeof(float8));
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
        elog(ERROR, "the second parameter's value range should be within the dimensions of the first parameter.");
    s2 = c1;
    if (d2 != n1)
    {
        for (uint32 i=0;i < d2;i++) s2 /= d1[i];
    }
    pg_tensor_shuffle(t1, s2, c1, (void*) p1);
    PG_RETURN_ARRAYTYPE_P(a1);
}

Datum array_binaryop(PG_FUNCTION_ARGS)
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
        pg_tensor_binaryop(t1, 1, c1, (void*) p1, (void*) p2);
    else if (strcasecmp(fn, "sub") == 0)
        pg_tensor_binaryop(t1, 2, c1, (void*) p1, (void*) p2);
    else if (strcasecmp(fn, "mul") == 0)
        pg_tensor_binaryop(t1, 3, c1, (void*) p1, (void*) p2);
    else if (strcasecmp(fn, "div") == 0)
        pg_tensor_binaryop(t1, 4, c1, (void*) p1, (void*) p2);

    PG_RETURN_ARRAYTYPE_P(a1);
}

Datum array_convolve(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3, *a5;
    char      *df, *pd, *p1, *p2;
    void      *p5;
    Oid        t1,  t2;
    int        n1, *d1, n2, *d2, n3, *d3, c3, c4, *p3, *p4;
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
        p3 = (int*)ARR_DATA_PTR(a3);
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
    if (a3 != NULL)
    {
        if (c3 != n5-1 || p3[c3-1] != 1)
            elog(ERROR, "strides shape does not meet conv%dd operation.", n5-2);
        for (uint32 i=0;i < n1-1;i++)
        {
            if (p3[i] <= 0 || p3[i] > d1[i+1])
                elog(ERROR, "strides shape does not meet conv%dd operation.", n1-2);
        }
    }
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
                if (d1[i] % p3[i-1] == 0)
                    s[i] = d1[i] / p3[i-1];
                else
                    s[i] = d1[i] / p3[i-1] + 1;
            }
        }
        c4 = (n5-1) * 2;
        p4 = (int *) palloc0(c4 * sizeof(int));
        for (uint32 i=0;i < n5-2;i++)
        {
            if (d2[i] > p3[i])
            {
                p4[2*i] = (d2[i] - 1) / 2;
                p4[2*i+1] = ((d2[i] - 1) / 2) + ((d2[i] - 1) % 2);
            }
            else
            {
                int32 _p_ = s[i+1] * p3[i] - d1[i+1];
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
                if ((d1[i] - d2[i-1] + 1) % p3[i-1] == 0)
                    s[i] = (d1[i] - d2[i-1] + 1) / p3[i-1];
                else
                    s[i] = (d1[i] - d2[i-1] + 1) / p3[i-1] + 1;
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
    pg_tensor_convolve(t1, (void*) p1, n1, d1, (void*) p2, d2, p3, p4, (void*) p5, d5);
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

Datum array_pool(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3, *a5;
    char      *fn, *df, *pd, *p1;
    int       *p2, *p3;
    void      *p5;
    Oid        t1;
    int        n1, *d1, n2, *d2, c2, n3, *d3, c3, c4, *p4;
    int        n5, *d5, *b5, c5 = 1, l5, s[6] = {0,0,0,0,0,0};
    instr_time s1,  s2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "pooling function name not specified.");
    fn = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (strcasecmp(fn, "max") != 0 && strcasecmp(fn, "avg") != 0)
        elog(ERROR, "\"%s\" is currently not supported in tensor pooling.", fn);
    if (PG_ARGISNULL(1))
        elog(ERROR, "data format not specified.");
    df = text_to_cstring(PG_GETARG_TEXT_P(1));
    if (strcasecmp(df, "NWC") != 0 && strcasecmp(df, "NHWC") != 0 && strcasecmp(df, "NDHWC") != 0)
        elog(ERROR, "\"%s\" is not supported in tensor pooling input.", df);
    if (PG_ARGISNULL(2)) PG_RETURN_NULL();
    if (PG_ARGISNULL(3))
        elog(ERROR, "pooling kernel sizes not specified.");
    a1 = PG_GETARG_ARRAYTYPE_P(2);
    a2 = PG_GETARG_ARRAYTYPE_P(3);
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    n2 = ARR_NDIM(a2);
    if (n2 != 1) elog(ERROR, "ksize shape must be one dimension.");
    d2 = ARR_DIMS(a2);
    p1 = ARR_DATA_PTR(a1);
    p2 = (int*)ARR_DATA_PTR(a2);
    c2 = ArrayGetNItems(n2, d2);
    if (PG_ARGISNULL(4))
    {
        a3 = NULL;
        n3 = 0;
        d3 = NULL;
        p3 = NULL;
    }
    else
    {
        a3 = PG_GETARG_ARRAYTYPE_P(4);
        n3 = ARR_NDIM(a3);
        if (n3 != 1) elog(ERROR, "strides shape must be one dimension.");
        d3 = ARR_DIMS(a3);
        c3 = ArrayGetNItems(n3, d3);
        p3 = (int32 *) ARR_DATA_PTR(a3);
    }
    if (PG_ARGISNULL(5))
        elog(ERROR, "padding type not specified.");
    else
    {
        pd = text_to_cstring(PG_GETARG_TEXT_P(5));
        if (strcasecmp(pd, "SAME") != 0 && strcasecmp(pd, "VALID") != 0)
            elog(ERROR, "\"%s\" is not supported in tensor pooling padding.", pd);
    }
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "input argument type must be float array type.");
    s[0] = d1[0];
    if (strcasecmp(df, "NWC") == 0)
        n5 = 3;
    else if (strcasecmp(df, "NHWC") == 0)
        n5 = 4;
    else if (strcasecmp(df, "NDHWC") == 0)
        n5 = 5;
    s[n5-1] = d1[n5-1];
    if (n1 != n5 || c2 != n5 || p2[0] != 1 || p2[c2-1] != 1)
        elog(ERROR, "input or ksize shape does not meet pool%dd operation.", n5-2);
    for (uint32 i=0;i < n5;i++)
        if (p2[i] > d1[i]) elog(ERROR, "input or ksize shape does not meet pool%dd operation.", n5-2);
    if (a3 != NULL)
    {
        if (c3 != n5 || p3[0] != 1 || p3[c3-1] != 1)
            elog(ERROR, "strides shape does not meet pool%dd operation.", n5-2);
        for (uint32 i=0;i < n1;i++)
        {
            if (p3[i] <= 0 || p3[i] > d1[i])
                elog(ERROR, "strides shape does not meet pool%dd operation.", n1-2);
        }
    }
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
                if (d1[i] % (p3[i]) == 0)
                    s[i] = d1[i] / p3[i];
                else
                    s[i] = d1[i] / p3[i] + 1;
            }
        }
        c4 = n5 * 2;
        p4 = (int *) palloc0(c4 * sizeof(int));
        for (uint32 i=1;i < n5-1;i++)
        {
            if (p2[i] > p3[i])
            {
                p4[2*i] = (p2[i] - 1) / 2;
                p4[2*i+1] = ((p2[i] - 1) / 2) + ((p2[i] - 1) % 2);
            }
            else
            {
                int32 _p_ = s[i] * p3[i] - d1[i];
                p4[2*i] = _p_ / 2;
                p4[2*i+1] = (_p_ / 2) + (_p_ % 2);
            }
        }
    }
    else
    {
        if (n3 == 0)
        {
            for (uint32 i=1;i < n5-1;i++) s[i] = d1[i] - p2[i] + 1;
        }
        else
        {
            for (uint32 i=1;i < n5-1;i++)
            {
                if ((d1[i] - p2[i] + 1) % (p3[i]) == 0)
                    s[i] = (d1[i] - p2[i] + 1) / p3[i];
                else
                    s[i] = (d1[i] - p2[i] + 1) / p3[i] + 1;
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
    if (strcasecmp(fn, "max") == 0)
        pg_tensor_pool(t1, 1, (void*) p1, n1, d1, p2, p3, p4, (void*) p5, d5);
    else if (strcasecmp(fn, "avg") == 0)
        pg_tensor_pool(t1, 2, (void*) p1, n1, d1, p2, p3, p4, (void*) p5, d5);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen pooling spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

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

Datum array_activate(PG_FUNCTION_ARGS)
{
    ArrayType *a1;
    char      *fn, *p1;
    Oid        t1;
    int        n1, c1, *d1;
    float4     g = 0;

    if (PG_ARGISNULL(0))
        elog(ERROR, "activate function name not specified.");
    fn = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (strcasecmp(fn, "sigmoid") != 0 && strcasecmp(fn, "tanh") != 0 && strcasecmp(fn, "relu") != 0 && strcasecmp(fn, "leaky relu") != 0 && strcasecmp(fn, "elu") != 0)
        elog(ERROR, "\"%s\" is currently not supported in tensor activation.", fn);
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    if (strcasecmp(fn, "leaky relu") == 0 || strcasecmp(fn, "elu") == 0)
    {
        if (PG_ARGISNULL(2))
            elog(ERROR, "gradient alpha not specified.");
        g = PG_GETARG_FLOAT4(2);
        if (g <= 0 || g >= 1)
            elog(ERROR, "gradient alpha value is unreasonable.");
    }
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "input argument type must be float array type.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    p1 = ARR_DATA_PTR(a1);
    c1 = ArrayGetNItems(n1, d1);

    if (strcasecmp(fn, "relu") == 0)
        pg_tensor_activate(t1, 1, c1, (void*) p1, g);
    else if (strcasecmp(fn, "sigmoid") == 0)
        pg_tensor_activate(t1, 2, c1, (void*) p1, g);
    else if (strcasecmp(fn, "tanh") == 0)
        pg_tensor_activate(t1, 3, c1, (void*) p1, g);
    else if (strcasecmp(fn, "leaky relu") == 0)
        pg_tensor_activate(t1, 4, c1, (void*) p1, g);
    else if (strcasecmp(fn, "elu") == 0)
        pg_tensor_activate(t1, 5, c1, (void*) p1, g);

    PG_RETURN_ARRAYTYPE_P(a1);
}

Datum array_dropout(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2;
    char      *p1;
    Oid        t1;
    float4     r2;
    int        n1, *d1, n2, *d2, c2;
    int       *p2,  s2;
    instr_time s0,  s1;

    if (PG_ARGISNULL(0)) PG_RETURN_NULL();
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    t1 = ARR_ELEMTYPE(a1);
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    p1 = ARR_DATA_PTR(a1);
    if (t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "input argument type must be float array type.");
    if (PG_ARGISNULL(1))
        elog(ERROR, "drop rate not specified.");
    r2 = PG_GETARG_FLOAT4(1);
    if (r2 < 0 || r2 >= 1) elog(ERROR, "drop rate is unreasonable.");
    if (PG_ARGISNULL(2))
    {
        a2 = NULL;
        n2 = 0;
        d2 = NULL;
        c2 = 0;
        p2 = NULL;
    }
    else
    {
        a2 = PG_GETARG_ARRAYTYPE_P(2);
        n2 = ARR_NDIM(a2);
        d2 = ARR_DIMS(a2);
        c2 = ArrayGetNItems(n2, d2);
        p2 = (int*)ARR_DATA_PTR(a2);
        if (n1 != c2) elog(ERROR, "noise shape is unreasonable.");
        for (int i=0;i < c2;i++)
        {
            if ((p2[i] <= 0) || (p2[i] > d1[i]) || (p2[i] != 1 && d1[i] % p2[i] != 0))
                elog(ERROR, "noise shape is unreasonable.");
        }
    }
    if (PG_ARGISNULL(3))
        elog(ERROR, "random seed not specified.");
    s2 = PG_GETARG_INT32(3);
    if (s2 < 0) elog(ERROR, "random seed is unreasonable.");

    INSTR_TIME_SET_CURRENT(s0);
    pg_tensor_dropout(t1, (void*) p1, n1, d1, r2, p2, s2);
    INSTR_TIME_SET_CURRENT(s1);
    INSTR_TIME_SUBTRACT(s1,s0);
    ereport(LOG,(errmsg("eigen dropout spend time %lu us", INSTR_TIME_GET_MICROSEC(s1))));

    PG_RETURN_ARRAYTYPE_P(a1);
}

Datum array_matmul(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3, *a4, *a5;
    char      *p1, *p2;
    Oid        t1,  t2;
    int32      n1,  n2,  c1,  c2;
    int32      n3,  n4,  c3,  c4;
    int32      ls,  rs,	 lz,  rz;
    int32     *d1, *d2, *d3, *d4, *p3;
    int32     *d5, *b5,  l5,  c5 = 1 ,  m5[6] = {0,0,0,0,0,0};
    bool      *p4;
    void      *v5;
    instr_time s1,  s2;

    if (PG_ARGISNULL(0)) PG_RETURN_NULL();
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    if (PG_ARGISNULL(2)) elog(ERROR, "matrix multiplication dimensions array not specified.");
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    a2 = PG_GETARG_ARRAYTYPE_P(1);
    a3 = PG_GETARG_ARRAYTYPE_P(2);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != INT2OID && t1 != INT4OID && t1 != INT8OID && t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "array argument type must be number type.");
    t2 = ARR_ELEMTYPE(a2);
    if (t2 != t1)
        elog(ERROR, "left and right matrix array must be same type.");
    n1 = ARR_NDIM(a1);
    n2 = ARR_NDIM(a2);
    if (n1 != n2)
        elog(ERROR, "left and right matrix array must have same dimensions.");
    d1 = ARR_DIMS(a1);
    d2 = ARR_DIMS(a2);
    n3 = ARR_NDIM(a3);
    if (n3 != 1) elog(ERROR, "matrix multiplication dimensions array must be 1 dimension.");
    d3 = ARR_DIMS(a3);
    c1 = ArrayGetNItems(n1, d1);
    c2 = ArrayGetNItems(n2, d2);
    if (c1 != c2)
        elog(ERROR, "the number of elements in the left and right matrix arrays must be consistent.");
    c3 = ArrayGetNItems(n3, d3);
    if (c3 != 2) elog(ERROR, "matrix multiplication dimensions array length must be 2.");
    p1 = ARR_DATA_PTR(a1);
    p2 = ARR_DATA_PTR(a2);
    p3 = (int32*) ARR_DATA_PTR(a3);
    for (int32 i=0;i < c3;i++)
    {
        if (p3[i] < 0 || p3[i] > n1-1)
            elog(ERROR, "matrix multiplication dimensions is unreasonable.");
    }
    if ((p3[0] != n1-2 || p3[1] != n1-1) && (p3[0] != n1-3 || p3[1] != n1-2))
        elog(ERROR, "matrix array or multiplication dimensions is unreasonable.");
    lz = d1[p3[1]];
    rz = d2[p3[0]];
    ls = d1[p3[0]];
    rs = d2[p3[1]];
    if (PG_ARGISNULL(3))
    {
        a4 = NULL;
        n4 = 0;
        d4 = NULL;
        c4 = 0;
        p4 = NULL;
    }
    else
    {
        a4 = PG_GETARG_ARRAYTYPE_P(3);
        n4 = ARR_NDIM(a4);
        if (n4 != 1) elog(ERROR, "transpose & conjugate configuration array must be 1 dimension.");
        d4 = ARR_DIMS(a4);
        c4 = ArrayGetNItems(n4, d4);
        if (c4 != 4) elog(ERROR, "transpose & conjugate configuration array length must be 4.");
        p4 = (bool*) ARR_DATA_PTR(a4);
        if (p4[0] || p4[2])
        {
            lz = d1[p3[0]];
            ls = d1[p3[1]];
        }
        if (p4[1] || p4[3])
        {
            rz = d2[p3[1]];
            rs = d2[p3[0]];
        }
    }
    for (int32 i=0;i < p3[0];i++)
    {
        if (d1[i] != d2[i])
            elog(ERROR, "left and right %s matrix array must have same batch sizes.", p3[0] == n1-3 ? "complex" : "real");
        m5[i] = d1[i];
    }
    if (lz != rz)
        elog(ERROR, "left and right %s matrix array contraction size incompatible.", p3[0] == n1-3 ? "complex" : "real");
    m5[p3[0]] = ls;
    m5[p3[1]] = rs;
    if (p3[0] == n1-3) m5[n1-1] = 2;
    for (int32 i=0;i < n1;i++) c5 *= m5[i];
    if (t1 == FLOAT4OID)
    {
        v5 = palloc(c5 * sizeof(float4));
        l5 = c5 * sizeof(float4) + ARR_OVERHEAD_NONULLS(n1);
    }
    else if (t1 == FLOAT8OID)
    {
        v5 = palloc(c5 * sizeof(float8));
        l5 = c5 * sizeof(float8) + ARR_OVERHEAD_NONULLS(n1);
    }
    else if (t1 == INT2OID)
    {
        v5 = palloc(c5 * sizeof(int16));
        l5 = c5 * sizeof(int16) + ARR_OVERHEAD_NONULLS(n1);
    }
    else if (t1 == INT4OID)
    {
        v5 = palloc(c5 * sizeof(int32));
        l5 = c5 * sizeof(int32) + ARR_OVERHEAD_NONULLS(n1);
    }
    else if (t1 == INT8OID)
    {
        v5 = palloc(c5 * sizeof(int64));
        l5 = c5 * sizeof(int64) + ARR_OVERHEAD_NONULLS(n1);
    }
    d5 = (int *) palloc(n1 * sizeof(int));
    b5 = (int *) palloc(n1 * sizeof(int));
    for (uint32 i=0;i < n1;i++)
    {
        d5[i] = m5[i];
        b5[i] = 1;
    }

    INSTR_TIME_SET_CURRENT(s1);
    pg_tensor_matmul(t1, p3[0], n1, (void*) p1, d1, (void*) p2, d2, p4, v5, d5);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen tensor matrix multiplication spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

    a5 = (ArrayType *) palloc0(l5);
    SET_VARSIZE(a5, l5);
    a5->ndim = n1;
    a5->dataoffset = 0;
    a5->elemtype = t1;
    memcpy(ARR_DIMS(a5)  , d5, n1 * sizeof(int));
    memcpy(ARR_LBOUND(a5), b5, n1 * sizeof(int));
    if (t1 == FLOAT4OID)
        memcpy(ARR_DATA_PTR(a5), v5, c5 * sizeof(float4));
    else if (t1 == FLOAT8OID)
        memcpy(ARR_DATA_PTR(a5), v5, c5 * sizeof(float8));
    else if (t1 == INT2OID)
        memcpy(ARR_DATA_PTR(a5), v5, c5 * sizeof(int16 ));
    else if (t1 == INT4OID)
        memcpy(ARR_DATA_PTR(a5), v5, c5 * sizeof(int32 ));
    else if (t1 == INT8OID)
        memcpy(ARR_DATA_PTR(a5), v5, c5 * sizeof(int64 ));
    pfree(v5);
    pfree(d5);
    pfree(b5);
    PG_RETURN_ARRAYTYPE_P(a5);
}

Datum array_softmax(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2;
    char      *p1;
    Oid        t1;
    int32     *d1, *b1, n1, c1, l2, ax;
    void      *v2;
    instr_time s1,  s2;

    if (PG_ARGISNULL(0)) PG_RETURN_NULL();
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != INT2OID && t1 != INT4OID && t1 != INT8OID && t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "array argument type must be number type.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    b1 = ARR_LBOUND(a1);
    c1 = ArrayGetNItems(n1, d1);
    p1 = ARR_DATA_PTR(a1);
    if (PG_ARGISNULL(1))
        ax = n1-1;
    else
        ax = PG_GETARG_INT32(1);
    if (ax < 0 || ax >= n1)
        elog(ERROR, "the axis to reduce sum across should be within the dimensions of the input tensor.");
    v2 = palloc(c1 * sizeof(float8));
    l2 = c1 * sizeof(float8) + ARR_OVERHEAD_NONULLS(n1);
    INSTR_TIME_SET_CURRENT(s1);
    pg_tensor_softmax(t1, (void*) p1, n1, d1, ax, v2);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen tensor matrix multiplication spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));
    a2 = (ArrayType *) palloc0(l2);
    SET_VARSIZE(a2, l2);
    a2->ndim = n1;
    a2->dataoffset = 0;
    a2->elemtype = FLOAT8OID;
    memcpy(ARR_DIMS(a2)  , d1, n1 * sizeof(int32));
    memcpy(ARR_LBOUND(a2), b1, n1 * sizeof(int32));
    memcpy(ARR_DATA_PTR(a2), v2, c1 * sizeof(float8));
    pfree(v2);
    PG_RETURN_ARRAYTYPE_P(a2);
}

Datum array_argpos(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2;
    char      *fn, *p1;
    Oid        t1;
    int32     *d1,  n1, c1, ax;
    int32     *d2, *b2, n2, c2, l2, j = 0;
    void      *v2;
    instr_time s1,  s2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "index reduce function name not specified.");
    fn = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (strcasecmp(fn, "argmax") != 0 && strcasecmp(fn, "argmin") != 0)
        elog(ERROR, "\"%s\" is currently not supported in tensor index reduce.", fn);
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    t1 = ARR_ELEMTYPE(a1);
    if (t1 != INT2OID && t1 != INT4OID && t1 != INT8OID && t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "array argument type must be number type.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    c1 = ArrayGetNItems(n1, d1);
    p1 = ARR_DATA_PTR(a1);
    if (PG_ARGISNULL(2))
        ax = 0;
    else
        ax = PG_GETARG_INT32(2);
    if (ax < 0 || ax >= n1)
        elog(ERROR, "the ax to reduce across should be within the dimensions of the input tensor.");
    n2 = n1 - 1;
    c2 = c1 / d1[ax];
    d2 = (int32 *) palloc(n2 * sizeof(int32));
    b2 = (int32 *) palloc(n2 * sizeof(int32));
    for (uint32 i=0;i < n1;i++)
    {
        if (ax == i) continue;
        d2[j] = d1[i]; b2[j] = 1; j++;
    }
    v2 = palloc(c2 * sizeof(int64));
    l2 = c2 * sizeof(int64) + ARR_OVERHEAD_NONULLS(n2);
    INSTR_TIME_SET_CURRENT(s1);
    if (strcasecmp(fn, "argmax") == 0)
        pg_tensor_argpos(t1, 1, p1, n1, d1, v2, ax);
    else if (strcasecmp(fn, "argmin") == 0)
        pg_tensor_argpos(t1, 2, p1, n1, d1, v2, ax);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen tensor matrix multiplication spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));
    a2 = (ArrayType *) palloc(l2);
    SET_VARSIZE(a2, l2);
    a2->ndim = n2;
    a2->dataoffset = 0;
    a2->elemtype = INT8OID;
    memcpy(ARR_DIMS(a2)  , d2, n2 * sizeof(int32));
    memcpy(ARR_LBOUND(a2), b2, n2 * sizeof(int32));
    memcpy(ARR_DATA_PTR(a2), v2, c2 * sizeof(int64));
    pfree(b2);
    pfree(d2);
    pfree(v2);
    PG_RETURN_ARRAYTYPE_P(a2);
}

Datum array_loss(PG_FUNCTION_ARGS)
{
    ArrayType *a1, *a2, *a3;
    Oid        t1,  t2;
    char      *fn, *p1, *p2;
    int32      n1, *d1,  c1;
    int32      n2, *d2,  c2, ax;
    int32      n3, *d3, *b3, c3, l3, j = 0;
    void      *v3;
    instr_time s1,  s2;

    if (PG_ARGISNULL(0))
        elog(ERROR, "loss function name not specified.");
    fn = text_to_cstring(PG_GETARG_TEXT_P(0));
    if (strcasecmp(fn, "MAE") != 0 && strcasecmp(fn, "MSE") != 0 && strcasecmp(fn, "CCE") != 0 && strcasecmp(fn, "SCE") != 0)
        elog(ERROR, "\"%s\" is currently not supported in tensor loss calculation.", fn);
    if (PG_ARGISNULL(1)) PG_RETURN_NULL();
    if (PG_ARGISNULL(2)) PG_RETURN_NULL();
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    a2 = PG_GETARG_ARRAYTYPE_P(2);
    t1 = ARR_ELEMTYPE(a1);
    t2 = ARR_ELEMTYPE(a2);
    if (t1 != FLOAT4OID && t1 != FLOAT8OID)
        elog(ERROR, "array argument type must be float point type.");
    if (t1 != t2)
        elog(ERROR, "true and predict values type must be same currently.");
    n1 = ARR_NDIM(a1);
    d1 = ARR_DIMS(a1);
    n2 = ARR_NDIM(a2);
    d2 = ARR_DIMS(a2);
    if (n1 != n2)
        elog(ERROR, "the shapes of the true and predict values must be the same.");
    c1 = ArrayGetNItems(n1, d1);
    c2 = ArrayGetNItems(n2, d2);
    if (c1 != c2)
        elog(ERROR, "the shapes of the true and predict values must be the same.");
    for (uint32 i=0;i < n1;i++)
    {
        if (d1[i] != d2[i])
            elog(ERROR, "the shapes of the true and predict values must be the same.");
    }
    p1 = ARR_DATA_PTR(a1);
    p2 = ARR_DATA_PTR(a2);
    if (PG_ARGISNULL(3))
        ax = n1-1;
    else
        ax = PG_GETARG_INT32(3);
    if (ax < 0 || ax >= n1)
        elog(ERROR, "the axis to reduce sum across should be within the dimensions of the input tensor.");
    n3 = n1 == 1 ? 1 : n1 - 1;
    c3 = c1 / d1[ax];
    d3 = (int32 *) palloc(n3 * sizeof(int32));
    b3 = (int32 *) palloc(n3 * sizeof(int32));
    for (uint32 i=0;i < n1;i++)
    {
        if (ax == i) continue;
        d3[j] = d1[i]; b3[j] = 1; j++;
    }
    v3 = palloc(c3 * sizeof(float8));
    l3 = c3 * sizeof(float8) + ARR_OVERHEAD_NONULLS(n3);

    INSTR_TIME_SET_CURRENT(s1);
    if (strcasecmp(fn, "MAE") == 0)
        pg_tensor_loss(t1, 1, (void*) p1, n1, d1, (void*) p2, (void*) v3, ax);
    else if (strcasecmp(fn, "MSE") == 0)
        pg_tensor_loss(t1, 2, (void*) p1, n1, d1, (void*) p2, (void*) v3, ax);
    else if (strcasecmp(fn, "CCE") == 0)
        pg_tensor_loss(t1, 3, (void*) p1, n1, d1, (void*) p2, (void*) v3, ax);
    else if (strcasecmp(fn, "SCE") == 0)
        pg_tensor_loss(t1, 4, (void*) p1, n1, d1, (void*) p2, (void*) v3, ax);
    INSTR_TIME_SET_CURRENT(s2);
    INSTR_TIME_SUBTRACT(s2,s1);
    ereport(LOG,(errmsg("eigen tensor matrix multiplication spend time %lu us", INSTR_TIME_GET_MICROSEC(s2))));

    a3 = (ArrayType *) palloc(l3);
    SET_VARSIZE(a3, l3);
    a3->ndim = n3;
    a3->dataoffset = 0;
    a3->elemtype = FLOAT8OID;
    memcpy(ARR_DIMS(a3)  , d3, n3 * sizeof(int32));
    memcpy(ARR_LBOUND(a3), b3, n3 * sizeof(int32));
    memcpy(ARR_DATA_PTR(a3), v3, c3 * sizeof(float8));
    pfree(b3);
    pfree(d3);
    pfree(v3);
    PG_RETURN_ARRAYTYPE_P(a3);
}
