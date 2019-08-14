import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
scatter_module = tf.load_op_library('./scatter_kernel.so')

n = 128
c = 16
h = 32
w = 32
nnz = 32

input = tf.constant(np.random.random((n,c,h,w)),dtype=tf.float32)
r_input = tf.reshape(input,[n,c,-1])
values, indices = tf.math.top_k(r_input,k=32)
print(input.get_shape())
output = scatter_module.nchw_scatter(values,indices,input)
print(output[0,0,0:10,:])
