import torch
import numpy as np

filter_x = 5
filter_y = 5
batch = 128
ic = 64
oc = 64
image_dim = 12

image = torch.Tensor(np.random.random((batch,ic,image_dim,image_dim)))
dy = torch.Tensor(np.random.random((batch,oc,image_dim,image_dim)))

NHWC_image = image.permute(0,2,3,1)
NHWC_dy = dy.permute(0,2,3,1)

reshaped_dy =