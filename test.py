import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

image = np.random.random((128,10,10,3))

row_idx = np.random.randint(5,size=(128,))
col_idx = np.random.randint(5,size=(128,))

start_idx = tf.stack([row_idx,col_idx],axis=1)
glimpses = tf.image.extract_glimpse(image, [5,5],tf.cast(start_idx, tf.float32), centered=False,normalized=False)

print(start_idx[0])
print(image[0,:,:,0])
print(image[0,row_idx[0]:row_idx[0]+5,col_idx[0]:col_idx[0]+5,0])
print(glimpses[0,:,:,0])
