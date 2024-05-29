# Model Notes

## 1.`model.py`

### [1.1 Sequential, ModelList, ModuleDict](https://blog.csdn.net/QLeelq/article/details/115208866)

- [x] `nn.Sequential` has implemented `forward` function internally, so it is no necessary to write a `forward` function manually.

`nn.ModuleList` and`nn.ModuleDict` don't have an internal `foward ` functio implemented.

- [x] `nn.Seuqential` needs to execute strictly in order, while the other two modules can be called in any order.

### 1.2 Sharing same weight parameters

Weight trying involves sharing the same weight parameters between different parts of the model, often seen between the `word embedding layer` and the `output layer in language model`

```python
self.transformer.wte.weight =  self.lm_head.weight
```

### 1.3 [How to Initialize Weights in Pytorch](#https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1)

<!-- TO be added -->

### 1.4 Special parameter initialization

<!-- TODO: To be added  -->

### 1.5 Difference between `state_dict()`, `named_parameters()` and `parameters()`

**Learnable Parameters**

- [x] Learnable parameters also known as model parameters, those are that participate back propagation calculation and constantly update during the training process
- [x] Variable created by using`nn.parameter.Parameter()` are learnable parameters

- [x] A characteristic of `nn.parameter.Parameter()` type parameters is that default to `requires_grad=True`. This means during training process, if back propagation is needed, this parameters must be used

**Non trainable parameters**

- [x] Non-trainable parameters don't participate in learning and are not updated by the optimizer meaning they don't need to involve in backpropagation

- [x] Non-trainable parameters will be registered in `self._buffers` via `Module.register_parameter()` where `self._buffers` is an `OrderDict`

**named_parameters()**

- [x] The `model.named_parameters()` method returns a generator that contains only the names and actual values of trainable parameters that can be updated by the optimizer. You can print the parameters names and values by iterating over its generator

- [x] The method can be used to change `require_grad` attributes of trainable parameters, all you freeze certain the updating of layer's parameters.

**parameters()**

- [x] The `model.parameters()` method returns a generator that only contains the actual trainable parameter that can be updated by the optimizer.
- [x] The model also can change `require_grad` attributes of trainable parameters, since it only provides the parameters without their corresponding names, it is not convenient as `model.named_parameters()` when you need to modify the `require_grad`

- [x] The `model.state_dict()` method returns a ordered dictionary that contains names and values of all the model's parameters including both `trainable and non-trainable` parameters.

### 1.6 [`weight decay` and `learning rate decay`](https://blog.csdn.net/program_developer/article/details/80867468)

**weight decay**

The purpose of L2 regularization is to decay the weights to smaller values, which helps reduce the problem of model overfitting to some extent.
Therefore the weight decay is also known as L2 regularization.

- [x] Function: Weight decay(L2 regularization) can prevent the model from overfitting
- [x] Reflection: The L2 regularization term make has the effect of making `weight` smaller, but why does make `weight` smaller prevent overfitting?
  - [x] From the perspective of model complexity: Smaller weight imply, in a sense, lower complexity of the network, which leads to better fitting of the data.In some practical applications, this has been validated, as the effect of L2 regularization is often better than of model without regularization.
  - [x] From the perspective of mathematical: During overfitting, the coefficients of the fitting function are very large, since overfitting requires the fitting function to account for every point, resulting in fluctuations in final fitting function. In every interval, the function values change dramatically. Since the values of the independent variable are vert greatly, only sufficient large coefficient can enure the derivative values are very large. Regularization constrains the norm of the parameters, preventing them from becoming too large, thereby reducing the likelihood of overfitting to some extent.

**Learning rate decay**

When training a model, it is common to encounter a situation where, after balancing the speed and loss, choosing a relatively appropriate learning rate, the training loss stops decreasing beyond a certain point. For example, the training loss might fluctuate between 0.7 and 0.9 and can not decrease anymore.

The basic idea of learning rate decay is that the learning rate gradually decreases as training process.

- [x] Linear Decay: halving the learning rate every 5 epochs
- [x] Exponential Decay: The learning rate automatically decay with an increase in number of iterations. For example, multiplying the learning rate by 0.9998 every epochs.

### 1.7 FLOP calculation

**Formula Explanation**

<!-- TODO: Figure out the source of 6 -->

- [x] `6 * N`: N represent the total number of parameters in the model
  - [x] In one forward pass and one backward pass, each parameter typically involves approximately 6 operations (this is an approximate value, including 2 multiplication and 4 additions/other operations)
  - [x] The factor 6 is an empirical value representing the total number of computational operations involving all parameters during one forward and backward pass

**Self Attention Part**

<!-- TODO: Figure out the source of 12 -->

- [x] `12 * L * H * Q * T` captures the computational complexity of self attention mechanism. Each head(`H`) in each layer(`L`) perform `Q * T` operation, collective involving about 12 operation

## 2. `Bench.py`

### 2.1 Steps

- [x] Define hyper parameters
- [x] Convert data to target device
  - [x] techniques:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
  # auto mix precision
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
  ```
- [x] Split the data and Load it into memory
  - [x] techniques:
  ```python
  # load large object from local path to the memory
  np.memmap(os.path.join(dir_path, 'train.bin'), dtype = np.unit16, mode='r')
  x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  ```

- [x] Instantiate the model and optimizers
  - [x] techniques:
  ```python
  # Use decay and momentum
  optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
  # Compile the model to speed up
  model = torch.compile(model)
  ```

- [x] Train the model and record the logs
  - [x] use profile:
  - [x] split the data twice 
    ```python
    X, Y = get_batch('train')
      for k in range(num_steps):
        with ctx:
          logits, loss = model(X, Y)
        X, Y = get_batch('train')
    ```


### 2.4 profile

- [x] tutorial: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
