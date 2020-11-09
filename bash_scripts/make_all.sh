#!/bin/bash
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') );TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# mag reg count should be 32 when compiling for 64 channels or below to force occupancy

# otherwise use more registers at the expense of occupancy
nvcc -std=c++11 -c -o approx_kernel1616.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=16,input_channels=16,ys_2=32,ys_3=32,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel1632.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=32,input_channels=16,ys_2=16,ys_3=16,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel3232.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=32,input_channels=32,ys_2=16,ys_3=16,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel6464.cu.o approx_kernel_NHWC_fixed_3.cu.cc  ${TF_CFLAGS[@]} -D ys_1=64,input_channels=64,ys_2=8,ys_3=8,TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"
nvcc -std=c++11 -c -o approx_kernel_flex.cu.o approx_kernel_NHWC_flexible.cu.cc  ${TF_CFLAGS[@]} -D TEST=0,PROFILE=0,GOOGLE_CUDA=1 -x cu -m64 -Xptxas -O3 -Xcompiler -fPIC -O3 --generate-line-info -use_fast_math -arch=sm_61 -Xptxas="-v"

g++ -std=c++11 -shared -o approx_kernel1616.so approx_kernel_NHWC.cc approx_kernel1616.cu.o -DNAME="TonyConvGrad1616" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel1632.so approx_kernel_NHWC.cc approx_kernel1632.cu.o -DNAME="TonyConvGrad1632" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel3232.so approx_kernel_NHWC.cc approx_kernel3232.cu.o -DNAME="TonyConvGrad3232" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel6464.so approx_kernel_NHWC.cc approx_kernel6464.cu.o -DNAME="TonyConvGrad6464" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}
g++ -std=c++11 -shared -o approx_kernel_flex.so approx_kernel_NHWC.cc approx_kernel_flex.cu.o -DNAME="TonyConvGradFlex" ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}

cp approx_kernel1616.so ../models/official/resnet/.
cp approx_kernel1632.so ../models/official/resnet/.
cp approx_kernel3232.so ../models/official/resnet/.
cp approx_kernel6464.so ../models/official/resnet/.
cp approx_kernel_flex.so ../models/official/resnet/.


#/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -IFreeImage/include  -m64 -o baseline.cu.o -c baseline.cu.cc -D IMAGE_DIM=$1 ${TF_CFLAGS[@]} -lcudnn -Xcompiler -fPIC -O3 -arch=sm_61
#g++ -std=c++11 -shared -o baseline.so baseline.cc baseline.cu.o -D IMAGE_DIM=$1 ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart -lcudnn ${TF_LFLAGS[@]} -g
