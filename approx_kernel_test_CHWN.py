from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
approx_module = tf.load_op_library('./approx_kernel_faster.so')

filter_x = 5
filter_y = 5
batch = 32
ic = 64

image = tf.constant(np.random.random((batch, 55, 55, ic)),dtype=tf.float32)
dy = tf.constant(np.random.random((batch, 50, 50, 64)),dtype=tf.float32)

CHWN_image = tf.transpose(image,perm=[3,1,2,0])
NCHW_dy = tf.transpose(dy,perm=[0,3,1,2])

for i in range(1):
	kernel_output = approx_module.tony_conv_grad(CHWN_image,NCHW_dy,1,filter_x,filter_y)
kernel_output_t = tf.transpose(kernel_output,perm=[2,3,1,0])

reshaped_dy = tf.reshape(dy,[-1, dy.shape[1] * dy.shape[2], dy.shape[3]])
max_idx = tf.argmax(tf.abs(reshaped_dy), axis=1)
scaling = tf.expand_dims(tf.expand_dims(tf.reduce_sum(reshaped_dy[:,:,0],axis=1,keepdims=True),axis=2),axis=3)
row_idx = max_idx // dy.get_shape().as_list()[1]
col_idx = max_idx % dy.get_shape().as_list()[1]

start_idx = tf.stack([row_idx,col_idx],axis=1)

i = 0

a = tf.expand_dims(tf.tile(tf.reshape(tf.range(batch), [-1, 1, 1]), [1, filter_x, filter_y]), 3)
b = tf.expand_dims(
    tf.map_fn(lambda x: tf.tile(tf.expand_dims(tf.range(x, x + filter_x), 1), [1, filter_y]),
              row_idx[:,i]),
    3)
c = tf.expand_dims(
    tf.map_fn(lambda x: tf.tile(tf.expand_dims(tf.range(x, x + filter_y), 0), [filter_x, 1]),
              col_idx[:,i]),
    3)
d = tf.squeeze(tf.stack([a, tf.cast(b, tf.int32), tf.cast(c, tf.int32)], axis=3),axis=4)
ref_result = tf.gather_nd(image,d)
ref_result = tf.reduce_mean(tf.multiply(scaling,ref_result),axis=0)



real_reference = []
for j in range(batch):
    slice = image[j,row_idx[j,0]:row_idx[j,0]+filter_x,col_idx[j,0]:col_idx[j,0]+filter_y,:] * tf.squeeze(scaling[j])
    real_reference.append(slice)


real_reference = tf.reduce_mean(tf.stack(real_reference,axis=0),axis=0)

print(scaling[0,0,0,0])
print(ref_result.shape,ref_result[:,:,0])
print(kernel_output.shape,kernel_output_t[:,:,0,0])
print("bump")
