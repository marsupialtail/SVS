#!/bin/bash
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') );TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# mag reg count should be 32 when compiling for 64 channels or below to force occupancy

# otherwise use more registers at the expense of occupancy
nvcc -std=c++11 -c -o approx_kernel6464.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=64,input_channels=64,ys_2=32,ys_3=32,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel128128.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=128,input_channels=128,ys_2=16,ys_3=16,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel256256.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=256,input_channels=256,ys_2=8,ys_3=8,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel512512.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=512,input_channels=512,ys_2=4,ys_3=4,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel512512_2.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=512,input_channels=512,ys_2=2,ys_3=2,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"

g++ -std=c++11 -shared -o approx_kernel6464.so approx_kernel_NHWC.cc approx_kernel6464.cu.o -DNAME="TonyConvGrad6464" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel128128.so approx_kernel_NHWC.cc approx_kernel128128.cu.o -DNAME="TonyConvGrad128128" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel256256.so approx_kernel_NHWC.cc approx_kernel256256.cu.o -DNAME="TonyConvGrad256256" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel512512.so approx_kernel_NHWC.cc approx_kernel512512.cu.o -DNAME="TonyConvGrad512512" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel512512_2.so approx_kernel_NHWC.cc approx_kernel512512_2.cu.o -DNAME="TonyConvGrad5125122" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}

cp approx_kernel6464.so ../cifar10-2/.
cp approx_kernel128128.so ../cifar10-2/.
cp approx_kernel256256.so ../cifar10-2/.
cp approx_kernel512512.so ../cifar10-2/.
cp approx_kernel512512_2.so ../cifar10-2/.


#/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -IFreeImage/include  -m64 -o baseline.cu.o -c baseline.cu.cc -D IMAGE_DIM=$1 ${TF_CFLAGS[@]} -lcudnn -Xcompiler -fPIC -O3 -arch=sm_61
#g++ -std=c++11 -shared -o baseline.so baseline.cc baseline.cu.o -D IMAGE_DIM=$1 ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart -lcudnn ${TF_LFLAGS[@]} -g
