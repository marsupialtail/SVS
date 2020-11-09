# Accelerating CNN Training Through Gradient Approximation

Ziheng Wang | Sree Harsha Nelaturu

The repository for the workshop paper presented at EMC^2, ISCA 2019.

To reproduce results for CIFAR10:

1. Ensure you have CUDA 8 and Tensorflow GPU Installed (Tested on TF 1.9)
2. In the CUDA_Kernels/ folder, run create_ops.sh.
3. Place the compiled approx_kernel2.so in the model_harness/models/
4. Run ```python cifar10_train.py```
