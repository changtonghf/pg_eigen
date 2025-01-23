MODULE_big = pg_eigen
EXTENSION = pg_eigen
DATA = pg_eigen--1.0.sql
PGFILEDESC = "pg_eigen - SQL wrapper for Eigen library"

SHLIB_LINK += -lstdc++
OBJS = pg_eigen.o eigen_wrap.o

NVCC := $(shell which nvcc)
ifdef NVCC
SHLIB_LINK += -L/usr/local/cuda/lib64 -lcudart
OBJS += eigen_cuda.o
NVCC += -ccbin /usr/bin/g++
NVCC_FLAGS = -w -g -G -m64 -std=c++11 -Xcompiler -fPIC -DEIGEN_USE_GPU
GENCODE_FLAGS += -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<
endif

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/pg_eigen
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif

ifeq ($(shell grep avx2 /proc/cpuinfo > /dev/null && echo 1), 1)
override CXXFLAGS += -mavx2
endif