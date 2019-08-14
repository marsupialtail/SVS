import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
add_one = tf.load_op_library('./cuda_op_kernel.so')
print(add_one.add_one(np.random.random((1000,1000,1000)).astype(np.int32)))
