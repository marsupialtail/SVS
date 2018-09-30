The speed of resnet-20 training with

- Every other layer custom grad, excluding size 1 filter. somewhere around 6.6 (2.0 on gtx 1080 not compil optimized for it though) (almost there!!!)
- If I transpose the input and use NHWC kernel, somewhere around 6.5. 
- Every other layer custom grad, empty kernel launch, excluding size 1 filter: 6.18
- Regular grad: 6.7 (2.1 on gtx1080)
-regular grad without all the switching overhead 6.6. The switching overhead is not significant. 

These results are interesting and suggest that we need to improve the speed of the kernel launch. However, 
there could be other factors that interfere with the consideration, for example custom kernel launch overhead

- Every other layer custom grad, excluding size 1 filter, minimal kernel launch (just initializing shared memory to 0), 6.2. This eliminates
doubts that I ahve some kernel launch overhead. Because this reserves the same amount of blocks. So any interference at the stream exectutor
level should be amde apparent

If we only ever use the patch at 1,1, the acc can get to 80%


Notes for approximating input gradient:
- if we just pass back tf.zeros, every other layer custom grad, excluding size 1 filter. we are going to have 5.774 with tony_grad for the filters. This means the potential saving for input_layer grad approx is around 1 second for every other layer, similar to the potential savings for the filter grad approx, which is expected. 

Notes for offloading sum and max_idx calc to Tensorflow
- Don't do it. It's too slow. (6.7 baseline with no calc in my kernel)

