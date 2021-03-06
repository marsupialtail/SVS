#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
<<<<<<< HEAD
import time
import sys
tf.enable_eager_execution()
approx_module = tf.load_op_library('approx_kernel_flex.so')
#baseline = tf.load_op_library('baseline.so')
filter_x = 3
filter_y = 3
batch = 128
ic = int(sys.argv[1])
oc = int(sys.argv[2])
image_dim = int(sys.argv[3])
image = tf.constant(np.random.random((batch, image_dim, image_dim, ic)),dtype=tf.float32)
dy = tf.constant(np.random.random((batch, image_dim, image_dim, oc)),dtype=tf.float32)
'''
padded_image = tf.pad(image, [[0, 0], [1, 2], [1, 2], [0,0]])
reshaped_dy = tf.reshape(dy,[-1, dy.shape[1] * dy.shape[2], dy.shape[3]])
max_idx = tf.argmax(tf.abs(reshaped_dy), axis=1)
scalings = tf.expand_dims(tf.expand_dims(tf.reduce_sum(reshaped_dy,axis=1,keepdims=False),axis=2),axis=3)
row_idx = max_idx // dy.get_shape().as_list()[1]
col_idx = max_idx % dy.get_shape().as_list()[1]
start_idx = tf.stack([row_idx,col_idx],axis=1)

i = 0
scaling = np.expand_dims(scalings[:,i,:,:],axis=2)

tf.enable_eager_execution()
approx_module = tf.load_op_library('./approx_kernel.so')

filter_x = 5
filter_y = 5
batch = 64
ic = 64

image = tf.constant(np.random.random((batch, 55, 55, ic)),dtype=tf.float32)
dy = tf.constant(np.random.random((batch, 50, 50, 64)),dtype=tf.float32)

NCHW_image = tf.transpose(image,perm=[0,3,1,2])
NCHW_dy = tf.transpose(dy,perm=[0,3,1,2])

for i in range(1):
	kernel_output = approx_module.tony_conv_grad(NCHW_image,NCHW_dy,1,filter_x,filter_y)
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
ref_result = tf.gather_nd(padded_image,d)
ref_result = tf.reduce_mean(tf.multiply(scaling,ref_result),axis=0)

NCHW_image = tf.transpose(image,perm=[0,3,1,2])
'''
NCHW_dy = tf.transpose(dy,perm=[0,3,1,2])

#base = baseline.cudnn_conv(NCHW_image,NCHW_dy)

kernel_output = approx_module.tony_conv_grad_flex(image,NCHW_dy,1,filter_x,filter_y)
with tf.device("gpu:0"):
  kernel_output_t = tf.transpose(kernel_output,perm=[1,2,3,0])

#start = time.time()
#tf.nn.conv2d_backprop_filter(input=NCHW_image, filter_sizes=[5,5,64,64], out_backprop=NCHW_dy,
#                                                 strides=[1,1,1,1],
#                                                 padding='VALID',data_format='NCHW')
#print("tensorflow time: " + str(time.time()-start))


#print(scaling[0,0,0,0])
#print(ref_result.shape,ref_result[:,:,0])
print(kernel_output.shape,kernel_output_t[:,:,0,0])

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
>>>>>>> 7a222eae67e371cd7ed234b6f510c059bbcd93e8
