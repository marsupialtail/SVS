#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import time

tf.enable_eager_execution()
approx_module = tf.load_op_library('./approx_kernel_CNHW.so')
baseline = tf.load_op_library('./baseline.so')
filter_x = 5
filter_y = 5
batch = 64
ic = 64
oc = 64
image_dim = 12

image = tf.constant(np.random.random((batch, ic,image_dim,image_dim)),dtype=tf.float32)
dy = tf.constant(np.random.random((batch, oc,image_dim,image_dim)),dtype=tf.float32)

CNHW_image = tf.transpose(image,perm=[1,0,2,3])

reshaped_dy = tf.reshape(dy,[dy.shape[0], dy.shape[1], dy.shape[2] * dy.shape[3]])
max_idx = tf.cast(tf.argmax(tf.abs(reshaped_dy), axis=2),tf.int32)
scalings = tf.expand_dims(tf.expand_dims(tf.reduce_sum(reshaped_dy,axis=2,keepdims=False),axis=2),axis=3)
print(max_idx.shape)
print(scalings.shape)
#row_idx = max_idx // image_dim
#col_idx = max_idx % image_dim

#start_idx = tf.stack([row_idx,col_idx],axis=1)

#i = 0
#scaling = np.expand_dims(scalings[:,i,:,:],axis=2)
#a = tf.expand_dims(tf.tile(tf.reshape(tf.range(batch), [-1, 1, 1]), [1, filter_x, filter_y]), 3)
#b = tf.expand_dims(
#    tf.map_fn(lambda x: tf.tile(tf.expand_dims(tf.range(x, x + filter_x), 1), [1, filter_y]),
#              row_idx[:,i]),
#    3)
#c = tf.expand_dims(
#    tf.map_fn(lambda x: tf.tile(tf.expand_dims(tf.range(x, x + filter_y), 0), [filter_x, 1]),
#              col_idx[:,i]),
#    3)
#d = tf.squeeze(tf.stack([a, tf.cast(b, tf.int32), tf.cast(c, tf.int32)], axis=3),axis=4)
#ref_result = tf.gather_nd(NHWC_image,d)
#ref_result = tf.reduce_mean(tf.multiply(scaling,ref_result),axis=0)


#testing =  tf.nn.conv2d_backprop_filter(input=image, filter_sizes=[filter_x,filter_y,ic,oc], out_backprop=dy,
#                                                 strides=[1,1,1,1],
#                                                 padding='SAME',data_format="NCHW")
#print(testing[:,:,0,0])
#sums = tf.squeeze(scalings)

#start = time.time()
kernel_output = approx_module.tony_conv_grad(CNHW_image,dy,max_idx,tf.squeeze(scalings))
kernel_output_t = tf.transpose(kernel_output,perm=[2,3,0,1])

#base = baseline.cudnn_conv(image,dy)
#print(base.shape)
#print("kernel time: " + str(time.time()-start))


#real_reference = []
#for j in range(batch):
#    slice = image[j,row_idx[j,0]:row_idx[j,0]+filter_x,col_idx[j,0]:col_idx[j,0]+filter_y,:] * tf.squeeze(scaling[j])
#    real_reference.append(slice)


#real_reference = tf.reduce_mean(tf.stack(real_reference,axis=0),axis=0)

#print(scaling[0,0,0,0])
#print(ref_result.shape,ref_result[:,:,0])
print(kernel_output_t.shape,kernel_output_t[:,:,0,0])
