#!/bin/bash
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') );TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# mag reg count should be 32 when compiling for 64 channels or below to force occupancy
# otherwise use more registers at the expense of occupancy
nvcc -std=c++11 -c -o scatter_kernel.cu.o scatter_kernel.cu.cc  ${TF_CFLAGS[@]} -D TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
g++ -std=c++11 -shared -o scatter_kernel.so scatter_kernel.cc scatter_kernel.cu.o -DNAME="TonyConvGradFlex" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
#/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -IFreeImage/include  -m64 -o baseline.cu.o -c baseline.cu.cc -D IMAGE_DIM=$1 ${TF_CFLAGS[@]} -lcudnn -Xcompiler -fPIC -O3 -arch=sm_61
#g++ -std=c++11 -shared -o baseline.so baseline.cc baseline.cu.o -D IMAGE_DIM=$1 ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart -lcudnn ${TF_LFLAGS[@]} -g
