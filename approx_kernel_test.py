from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
approx_module = tf.load_op_library('./approx_kernel.so')

filter_x = 5
filter_y = 5

image = tf.constant(np.random.random((128, 10, 10, 3)),dtype=tf.float32)
dy = tf.constant(np.random.random((128, 10, 10, 64)),dtype=tf.float32)

reshaped_dy = tf.reshape(dy,[-1, dy.shape[1] * dy.shape[2], dy.shape[3]])
max_idx = tf.argmax(tf.abs(reshaped_dy), axis=1)
scaling = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(reshaped_dy,axis=2,keepdims=True),axis=1,keepdims=True),axis=3)
row_idx = max_idx // dy.get_shape().as_list()[1]
col_idx = max_idx % dy.get_shape().as_list()[1]

start_idx = tf.stack([row_idx,col_idx],axis=1)

i = 0

a = tf.expand_dims(tf.tile(tf.reshape(tf.range(128), [-1, 1, 1]), [1, filter_x, filter_y]), 3)
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

NCHW_image = tf.transpose(image,perm=[0,3,1,2])
NCHW_dy = tf.transpose(dy,perm=[0,3,1,2])

kernel_output = approx_module.tony_conv_grad(NCHW_image,NCHW_dy)
kernel_output = tf.transpose(kernel_output,perm=[2,3,1,0])

print(ref_result[:,:,0])
print(kernel_output[:,:,0,0])
