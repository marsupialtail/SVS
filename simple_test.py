import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os

os.system("TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') );TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )")
os.system("TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )")
os.system("nvcc -std=c++11 -c -o approx_kernel.cu.o approx_kernel.cu.cc  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -G -g")
os.system("g++ -std=c++11 -shared -o approx_kernel.so approx_kernel.cc approx_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}")

approx_module = tf.load_op_library('./approx_kernel.so')

a = np.ones((10,3,20,20)).astype(np.float32)
a = tf.constant(a,dtype=tf.float32)
b = approx_module.tony_conv_grad(a,a)

print(b)

