#!/bin/bash
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') );TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -IFreeImage/include  -m64 -o baseline.cu.o -c baseline.cu.cc  ${TF_CFLAGS[@]} -D IC=$1,OC=$2,IMAGE_DIM=$3 -lcudnn -Xcompiler -fPIC -O3 -arch=sm_61
g++ -std=c++11 -shared -o baseline.so baseline.cc baseline.cu.o ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart -lcudnn ${TF_LFLAGS[@]} -g
