## Issues Faced and the fixes we applied to solve them

### Issue - 1 
> RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [20, 50]], which is output 0 of AsStridedBackward0, is at version 93; expected version 92 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).

When we were trying to compute back gradients and as well as modifying the tensors in place we got errors for in place modifications, we fixed it by appending the h_t to a list and then stacking them at the end. 

### Issue - 2

> RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

For this the logits were computed per slice in the loop and we were stacking them, which led to decoupling of stacked h tensor and loss computation. but pytorch needs the stacked h to be a part of the loss graph. So we fixed it by computing the logits at once after stacking whole h.
