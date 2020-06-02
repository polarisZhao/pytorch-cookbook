### 1. numpy array 和 Tensor(CPU & GPU)

~~~shell
>>> import torch
>>> import numpy as np
>>> a = np.ones(5)
>>> a
array([1., 1., 1., 1., 1.])
>>> b = torch.from_numpy(a)     # numpy array-> CPU Tensor
>>> b 
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
>>> y = y.cuda()     # CPU Tensor -> GPU Tensor
>>> y
tensor([1., 1., 1., 1., 1.], device='cuda:0', dtype=torch.float64)
>>> y = y.cpu()  # GPU Tensor-> CPU Tensor
>>> y
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
>>> y = y.numpy()  # CPU Tensor -> numpy array
>>> y
array([1., 1., 1., 1., 1.])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> y = torch.from_numpy(y)
>>> y.to(device) # 这里 x.to(device) 等价于 x.cuda()
tensor([1., 1., 1., 1., 1.], device='cuda:0', dtype=torch.float64)
~~~

索引、 view 是不会开辟新内存的，而像 y = x + y 这样的运算是会新开内存的，然后将 y 指向新内存。



### 2. Variable 和　Tensor (require_grad=True)

​    Pytorch 0.4 之前的模式为:　**Tensor 没有梯度计算，加上梯度更新等操作后可以变为Variable**.  Pytorch0.4 将 Tensor 和Variable 合并。默认 Tensor 的 require_grad 为 false，可以通过修改 requires_grad 来为其添加梯度更新操作。

~~~python
>>> y
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)  
>>> y.requires_grad
False
>>> y.requires_grad = True
>>> y
tensor([1., 1., 1., 1., 1.], dtype=torch.float64, requires_grad=True)
~~~



### 3. detach 和　with torch.no_grad()

一个比较好的 detach和 torch.no_grad区别的解释:

>**`detach()` detaches the output from the computationnal graph. So no gradient will be backproped along this variable.**
>
>**`torch.no_grad` says that no operation should build the graph.**
>
>**The difference is that one refers to only a given variable on which it’s called. The other affects all operations taking place within the with statement.**

`detach()` 将一个变量从计算图中分离出来，也没有了相关的梯度更新。`torch.no_grad()`只是说明该操作没必要建图。不同之处在于，前者只能指定一个给定的变量，后者 则会影响影响在 with 语句中发生的所有操作。

 

### 4. model.eval()　和 torch.no_grad()

>**These two have different goals:**
>
>- `model.eval()` will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
>
>- `torch.no_grad()` impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).



### 5. xx.data 和 xx.detach()

​      在 0.4.0 版本之前,  .data 的语义是 获取 Variable 的 内部 Tensor, 在 0.4.0 版本将 Variable 和 Tensor merge 之后,  `.data` 和之前有类似的语义， 也是内部的 Tensor 的概念。`x.data` 与 `x.detach()` 返回的 tensor 有相同的地方, 也有不同的地方:

**相同:**

- 都和 x 共享同一块数据
- 都和 x 的 计算历史无关
- requires_grad = False

**不同:**

- y= x.data 在某些情况下不安全, 某些情况, 指的就是上述 inplace operation 的第二种情况, 所以, release note 中指出, 如果想要 detach 的效果的话, 还是 detach() 安全一些.

~~~python
>>> import torch
>>> x = torch.FloatTensor([[1., 2.]])
>>> w1 = torch.FloatTensor([[2.], [1.]])
>>> w2 = torch.FloatTensor([3.])
>>> w1.requires_grad = True
>>> w2.requires_grad = True
>>> d = torch.matmul(x, w1)
>>> d_ = d.data
>>> f = torch.matmul(d, w2)
>>> d_[:] = 1
>>> f.backward()
~~~

**如果需要获取其值，可以使用  xx.cpu().numpy() 或者 xx.cpu().detach().numpy() 然后进行操作，不建议再使用 volatile和  xx.data操作。**



### 6. ToTensor & ToPILImage 各自都做了什么?

**ToTensor:**

- 取值范围：   [0, 255]  -->  [0, 1.0]

- NHWC  --> NCHW

- PILImage  --> FloatTensor

**ToPILImage:**

- 取值范围:  [0, 1.0] -->  [0, 255]

- NCHW --> NHWC

- 类型: FloatTensor -> numpy Uint8 -> PILImage