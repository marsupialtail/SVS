The speed of resnet-20 training with

- Every other layer custom grad, excluding size 1 filter. somewhere around 9
- Every other layer custom grad, empty kernel launch, excluding size 1 filter: 6.18
- Regular grad: 6.8

These results are interesting and suggest that we need to improve the speed of the kernel launch. However, 
there could be other factors that interfere with the consideration, for example custom kernel launch overhead

- Every other layer custom grad, excluding size 1 filter, minimal kernel launch (just initializing shared memory to 0), 6.2. This eliminates
doubts that I ahve some kernel launch overhead. Because this reserves the same amount of blocks. So any interference at the stream exectutor
level should be amde apparent

- No max calculation. 7.2 

If we only ever use the patch at 1,1, the acc can get to 80%
