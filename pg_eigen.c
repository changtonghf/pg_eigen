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

PG_FUNCTION_INFO_V1(array_reduce);
PG_FUNCTION_INFO_V1(array_fft);

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
