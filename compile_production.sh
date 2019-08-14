#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') );TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
nvcc -std=c++11 -c -o approx_kernel_production.cu.o approx_kernel_production.cu.cc  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O3
g++ -std=c++11 -shared -o approx_kernel_production.so approx_kernel.cc approx_kernel_production.cu.o ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
