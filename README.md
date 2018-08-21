# Support Vector Ssampling

Follow the following steps to compile the kernel to a linked Tensorflow library.

1. Install CUDA
2. Install Tensorflow 1.9 (as long as it supports custom gradient, so 1.7 and above I think?)
3. TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
4. TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
5. nvcc -std=c++11 -c -o approx_kernel.cu.o approx_kernel.cu.cc  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -G -g
(nvcc comes with CUDA. Note I am compiling with debug flags)
6. g++ -std=c++11 -shared -o approx_kernel.so approx_kernel.cc approx_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -L$CUDA_HOME/lib64 -lcudart ${TF_LFLAGS[@]}

Now you should see approx_kernel.so. 

Run this simple test:
```bash
import tensorflow as tf
approx_module = tf.load_op_library('./approx_kernel.so')

a = np.ones((10,3,20,20)).astype(np.float32)
a = tf.constant(a,dtype=tf.float32)
b = approx_module.tony_conv_grad(a,a)

print(b)
```


Follow the following steps to test the kernel on Resnet on CIFAR10. If you are going to do this, I recommend removing all the print statements in the .cu.cc file.

1. Download tensorflow/models/official/resnet
2. Make sure the shared library file that you compiled earlier approx_kernel.so is in the resnet folder. 
3. Replace resnet_model.py in the directory with resnet_model.py in the repo
4. Run python cifar10_main.py with your desired options. 

Common problem: if key not found in checkpoint, rm /tmp/cifar10_model is that's where you keep your checkpoint files. This is caused by changing resnet size. 
