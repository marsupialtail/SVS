"""
Demostrating how to compute the gradients for convolution with:
    tf.nn.conv2d
    tf.nn.conv2d_backprop_input
    tf.nn.conv2d_backprop_filter
    tf.nn.conv2d_transpose
This is the scripts for this answer: https://stackoverflow.com/a/44350789/1255535
"""

import tensorflow as tf
import numpy as np
import scipy.signal


def tf_rot180(w):
    """
    Roate by 180 degrees
    """
    return tf.reverse(w, axis=[0, 1])


def tf_pad_to_full_conv2d(x, w_size):
    """
    Pad x, such that using a 'VALID' convolution in tensorflow is the same
    as using a 'FULL' convolution. See
    http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d
    for description of 'FULL' convolution.
    """
    return tf.pad(x, [[0, 0],
                      [w_size - 1, w_size - 1],
                      [w_size - 1, w_size - 1],
                      [0, 0]])


def tf_NHWC_to_HWIO(out):
    """
    Converts [batch, in_height, in_width, in_channels]
    to       [filter_height, filter_width, in_channels, out_channels]
    """
    return tf.transpose(out, perm=[1, 2, 0, 3])

@tf.custom_gradient
def tony_conv(input,filter):

    strides = (1,1,1,1)
    padding = 'SAME'

    def grad(dy):

        reshaped_dy = tf.reshape(tf.transpose(dy,[0,3,1,2]),[dy.shape[0],dy.shape[1]*dy.shape[2],dy.shape[3]])
        max_idx = tf.argmax(reshaped_dy,axis=1)
        row_idx = max_idx // dy.shape[1]
        col_idx = max_idx % dy.shape[1]
        slices = []
        for i in range(filter.shape[3]):
            start_idx = tf.stack([row_idx[:,i],col_idx[:,i]],axis=1)
            glimpses = tf.image.extract_glimpse(input,[filter.shape[0],filter.shape[1]],tf.cast(start_idx,tf.float32),centered=False,normalized=False)
            slices.append(tf.reduce_sum(glimpses,axis=0))

        return [tf.nn.conv2d_backprop_input(input_sizes =input.shape,filter=filter,out_backprop=dy,strides=strides,padding=padding),
         tf.stack(slices,axis=3)]

    return tf.nn.conv2d(input,filter,strides=strides,padding='SAME'), grad

# sizes, fixed strides, in_channel, out_channel be 1 for now
x_size = 4
w_size = 3  # use an odd number here
x_shape = (1, x_size, x_size, 1)
w_shape = (w_size, w_size, 1, 1)
out_shape = (1, x_size - w_size + 1, x_size - w_size + 1, 1)
strides = (1, 1, 1, 1)

# numpy value
x_np = np.random.randint(10, size=x_shape)
w_np = np.random.randint(10, size=w_shape)

# tf forward
x = tf.constant(x_np, dtype=tf.float32)
w = tf.constant(w_np, dtype=tf.float32)
#out = tf.nn.conv2d(input=x, filter=w, strides=strides, padding='VALID')
out = tony_conv(x, w)


# tf backward
d_x = tf.gradients(out, x)[0]

with tf.Session() as sess:
    print(sess.run(out))
