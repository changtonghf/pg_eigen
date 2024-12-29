CREATE FUNCTION array_reduce(text,   int2[], int[]) RETURNS   int2[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_reduce(text,   int4[], int[]) RETURNS   int4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_reduce(text,   int8[], int[]) RETURNS   int8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_reduce(text, float4[], int[]) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_reduce(text, float8[], int[]) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;

CREATE FUNCTION array_fft(boolean, float4[], int[]) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_fft(boolean, float8[], int[]) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;

CREATE FUNCTION array_random(text, int4[], float8, float8, int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION array_shuffle(  int2[], int) RETURNS   int2[] AS 'MODULE_PATHNAME' LANGUAGE C;
CREATE FUNCTION array_shuffle(  int4[], int) RETURNS   int4[] AS 'MODULE_PATHNAME' LANGUAGE C;
CREATE FUNCTION array_shuffle(  int8[], int) RETURNS   int8[] AS 'MODULE_PATHNAME' LANGUAGE C;
CREATE FUNCTION array_shuffle(float4[], int) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C;
CREATE FUNCTION array_shuffle(float8[], int) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION array_binaryop(text,   int2[],   int2[]) RETURNS   int2[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_binaryop(text,   int4[],   int4[]) RETURNS   int4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_binaryop(text,   int8[],   int8[]) RETURNS   int8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_binaryop(text, float4[], float4[]) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_binaryop(text, float8[], float8[]) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;

CREATE FUNCTION array_convolve(text, float4[], float4[], int4[], text) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_convolve(text, float8[], float8[], int4[], text) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;

CREATE FUNCTION array_pool(text, text, float4[], int4[], int4[], text) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_pool(text, text, float8[], int4[], int4[], text) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;

CREATE FUNCTION array_activate(text, float4[], float4) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_activate(text, float8[], float4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;

CREATE FUNCTION array_dropout(float4[], float4, int4[], int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C;
CREATE FUNCTION array_dropout(float8[], float4, int4[], int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION array_matmul(  int2[],   int2[], int4[], boolean[]) RETURNS   int2[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_matmul(  int4[],   int4[], int4[], boolean[]) RETURNS   int4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_matmul(  int8[],   int8[], int4[], boolean[]) RETURNS   int8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_matmul(float4[], float4[], int4[], boolean[]) RETURNS float4[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_matmul(float8[], float8[], int4[], boolean[]) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;

CREATE FUNCTION array_softmax(  int2[], int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_softmax(  int4[], int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_softmax(  int8[], int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_softmax(float4[], int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
CREATE FUNCTION array_softmax(float8[], int4) RETURNS float8[] AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE;
