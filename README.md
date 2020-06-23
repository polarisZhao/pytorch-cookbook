# PyTorch Cookbook

[TOC]

## 一. Basic concept [alpha]

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

`model.eval()`和`torch.no_grad()`的区别在于，`model.eval()`是将网络切换为测试状态，例如BN和随机失活（dropout）在训练和测试阶段使用不同的计算方法。`torch.no_grad()`是关闭PyTorch张量的自动求导机制，以减少存储使用和加速计算，得到的结果无法进行`loss.backward()`

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

~~~python
# PIL.Image -> torch.Tensor.
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))
    ).permute(2, 0, 1).float() / 255
#　等价于
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) 
~~~

**ToPILImage:**

- 取值范围:  [0, 1.0] -->  [0, 255]
- NCHW --> NHWC
- 类型: FloatTensor -> numpy Uint8 -> PILImage

~~~python
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor * 255, min=0, max=255
    ).byte().permute(1, 2, 0).cpu().numpy())
#　等价于
image = torchvision.transforms.functional.to_pil_image(tensor) 
~~~

### 7. torch.nn.xxx 与 torch.nn.functional.xxx

建议统一使用　`torch.nn.xxx`　模块，`torch.functional.xxx` 可能会在下一个版本中去掉。

`torch.nn`　模块和　`torch.nn.functional`　的区别在于，`torch.nn`　模块在计算时底层调用了`torch.nn.functional`，但　`torch.nn`　模块包括该层参数，还可以应对训练和测试两种网络状态。使用　`torch.nn.functional`　时要注意网络状态，如:

~~~python
def forward(self, x):
    ...
    x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
~~~



## 二. Pytorch API [alpha]

### 1. import torch

import & vision

~~~python
import torch 
print(torch.__version__)
~~~

### 2. Tensor type 🌟

Pytorch 给出了 9 种 CPU Tensor 类型和 9 种 GPU Tensor 类型。Pytorch 中默认的数据类型是 torch.FloatTensor, 即 torch.Tensor 等同于 torch.FloatTensor。

| Data type                | dtype                         | CPU tensor         | GPU tensor              |
| ------------------------ | ----------------------------- | ------------------ | ----------------------- |
| 32-bit floating point    | torch.float32 or torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64-bit floating point    | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point    | torch.float16 or torch.half   | torch.HalfTensor   | torch.cuda.HalfTensor   |
| 8-bit integer (unsigned) | torch.uint8                   | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8-bit integer (signed)   | torch.int8                    | torch.CharTensor   | torch.cuda.CharTensor   |
| 16-bit integer (signed)  | torch.int16 or torch.short    | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32-bit integer (signed)  | torch.int32 or torch.int      | torch.IntTensor    | torch.cuda.IntTensor    |
| 64-bit integer (signed)  | torch.int64 or torch.long     | torch.LongTensor   | torch.cuda.LongTensor   |
| Boolean                  | torch.bool                    | torch.BoolTensor   | torch.cuda.BoolTensor   |

##### 设置默认Tensor 类型

Pytorch 可以通过 `set_default_tensor_type` 函数**设置默认使用的Tensor类型**， 在局部使用完后如果需要其他类型，则还需要重新设置会所需的类型 

~~~
torch.set_default_tensor_type('torch.DoubleTensor')
~~~

##### CPU/GPU 互转

CPU Tensor 和 GPU Tensor 的区别在于， 前者存储在内存中，而后者存储在显存中。两者之间的转换可以通过 `.cpu()`、`.cuda()`和 `.to(device)` 来完成  ※

~~~python
>>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
>>> a = torch.rand(2,3)
>>> a = a.cuda() # CPU -> GPU
>>> a.type()
'torch.cuda.FloatTensor'
>>> a = a.cpu() # GPU -> CPU
>>> a.type()
'torch.FloatTensor'
>>> a = a.to(device) # CPU <->  GPU
>>> a.type()
'torch.cuda.FloatTensor'
~~~

##### 判定 Tensor 类型的几种方式: ※

~~~python
>>> a
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], device='cuda:0')
>>> a.is_cuda  # 可以显示是否在显存中
True
>>> a.dtype  # Tensor 内部data的类型
torch.float32
>>> a.type()
'torch.cuda.FloatTensor'  # 可以直接显示Tensor类型 = is_cuda + dtype
~~~

##### 类型转换

~~~python
>>> a
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], device='cuda:0')
>>> a.type(torch.DoubleTensor)   # 使用 type() 函数进行转换
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], dtype=torch.float64)
>>> a = a.double()  # 直接使用 int()、long() 、float() 、和 double() 等直接进行数据类型转换进行
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], device='cuda:0', dtype=torch.float64)
>>> b = torch.randn(4,5)
>>> b.type_as(a)  # 使用 type_as 函数并不需要明确具体是哪种类型
tensor([[ 0.2129,  0.1877, -0.0626,  0.4607, -1.0375],
        [ 0.7222, -0.3502,  0.1288,  0.6786,  0.5062],
        [-0.4956, -0.0793,  0.7590, -1.0932, -0.1084],
        [-2.2198,  0.3827,  0.2735,  0.5642,  0.6771]], device='cuda:0',
       dtype=torch.float64)
~~~

##### numpy array 与　torch Tensor　互转

~~~python
torch.Tensor与np.ndarray转换
# torch.Tensor -> np.ndarray.
ndarray = tensor.cpu().numpy()

# np.ndarray -> torch.Tensor.
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float()  # If ndarray has negative stride
~~~

##### Tensor 相关信息获取

~~~python
torch.size()/torch.shape   # 两者等价， 返回t的形状, 可以使用 x.size()[1] 或 x.size(1) 查看列数
torch.numel() / torch.nelement()  # 两者等价, t中元素总个数
a.item()  # 取出单个tensor的值
tensor.dim()  # 维度
~~~

### 3. Tensor Create

##### 最基本的Tensor创建方式

~~~python
troch.Tensor(2, 2) # 会使用默认的类型创建 Tensor, 可以通过 torch.set_default_tensor_type('torch.DoubleTensor') 进行修改
torch.DoubleTensor(2, 2) # 指定类型创建 Tensor

torch.Tensor([[1, 2], [3, 4]])  # 通过 list 创建 Tensor          将 Tensor转换为list可以使用: t.tolist():
torch.from_numpy(np.array([2, 3.3]) ) # 通过 numpy array 创建 tensor
~~~

##### 确定初始值的方式创建

~~~python
torch.ones(sizes)  # 全 1 Tensor     
torch.zeros(sizes)  # 全 0 Tensor
torch.eye(sizes)  # 对角线为1，不要求行列一致
torch.full(sizes, value) # 指定 value
~~~

##### 分布

~~~python
torch.rand(sizes)  # 均匀分布   
torch.randn(sizes)   # 标准分布
# 正态分布: 返回一个张量，包含从给定参数 means,std 的离散正态分布中抽取随机数。 均值 means是一个张量，包含每个输出元素相关的正态分布的均值 -> 以此张量的均值作为均值
# std是一个张量，包含每个输出元素相关的正态分布的标准差 -> 以此张量的标准差作为标准差。 均值和标准差的形状不须匹配，但每个张量的元素个数须相同
torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([-0.1987,  3.1957,  3.5459,  2.8150,  5.5398,  5.6116,  7.5512,  7.8650,
         9.3151, 10.1827])
torch.uniform(from,to) # 均匀分布 

torch.arange(s, e, steps)  # 从s到e，步长为step
torch.linspace(s, e, num)   # 从s到e,均匀切分为 num 份, 注意linespace和arange的区别，前者的最后一个参数是生成的Tensor中元素的数量，而后者的最后一个参数是步长。
torch.randperm(m) # 0 到 m-1 的随机序列
# --> shuffle 操作
tensor[torch.randperm(tensor.size(0))] 
~~~

##### 复制

Pytorch 有几种不同的复制方式，注意区分

| Operation             | New/Shared memory | Still in computation graph |
| --------------------- | ----------------- | -------------------------- |
| tensor.clone()        | New               | Yes                        |
| tensor.detach()       | Shared            | No                         |
| tensor.detach.clone() | New               | No                         |

### 4. 索引、比较、排序

##### 索引操作

~~~python
a.item() #　从只包含一个元素的张量中提取值

a[row, column]   # row 行， cloumn 列
a[index]   # 第index 行
a[:,index]   # 第 index 列

a[0, -1]  # 第零行， 最后一个元素
a[:index]  # 前 index 行
a[:row, 0:1]  # 前 row 行， 0和1列

a[a>1]  # 选择 a > 1的元素， 等价于 a.masked_select(a>1)
torch.nonzero(a) # 选择非零元素的坐标，并返回
a.clamp(x, y)  # 对 Tensor 元素进行限制， 小于x用x代替， 大于y用y代替
torch.where(condition, x, y)  # 满足condition 的位置输出x， 否则输出y
>>> a
tensor([[ 6., -2.],
        [ 8.,  0.]])
>>> torch.where(a > 1, torch.full_like(a, 1), a)  # 大于1 的部分直接用1代替， 其他保留原值
tensor([[ 1., -2.],
        [ 1.,  0.]])

#　得到非零元素
torch.nonzero(tensor)               # 非零元素的索引
torch.nonzero(tensor == 0)          # 零元素的索引
torch.nonzero(tensor).size(0)       # 非零元素的个数
torch.nonzero(tensor == 0).size(0)  # 零元素的个数
~~~

##### 比较操作

~~~python
gt >    lt <     ge >=     le <=   eq ==    ne != 
topk(input, k) -> (Tensor, LongTensor)
sort(input) -> (Tensor, LongTensor)
max/min => max(tensor)      max(tensor, dim)    max(tensor1, tensor2)
~~~

sort 函数接受两个参数, 其中 参数 0 为按照行排序、1为按照列排序: True 为降序， False 为升序， 返回值有两个， 第一个是排序结果， 第二个是排序序号

~~~python
>>> import torch
>>> a = torch.randn(3, 3)
>>> a
tensor([[-1.8500, -0.2005,  1.4475],
        [-1.7795, -0.4968, -1.8965],
        [ 0.5798, -0.1554,  1.6395]])
>>> a.sort(0, True)[0] 
tensor([[ 0.5798, -0.1554,  1.6395],
        [-1.7795, -0.2005,  1.4475],
        [-1.8500, -0.4968, -1.8965]])
>>> a.sort(0, True)[1]
tensor([[2, 2, 2],
        [1, 0, 0],
        [0, 1, 1]])
>>> a.sort(1, True)[1]
tensor([[2, 1, 0],
        [1, 0, 2],
        [2, 0, 1]])
>>> a.sort(1, True)[0]
tensor([[ 1.4475, -0.2005, -1.8500],
        [-0.4968, -1.7795, -1.8965],
        [ 1.6395,  0.5798, -0.1554]])
~~~

### 5. Element-wise 和 归并操作

Element-wise：输出的 Tensor 形状与原始的形状一致

~~~python
abs / sqrt / div / exp / fmod / log / pow...
cos / sin / asin / atan2 / cosh...
ceil / round / floor / trunc
clamp(input, min, max)
sigmoid / tanh...
~~~

归并操作：输出的 Tensor 形状小于原始的 Tensor形状

~~~python
mean/sum/median/mode   # 均值/和/ 中位数/众数
norm/dist  # 范数/距离
std/var  # 标准差/方差
cumsum/cumprd # 累加/累乘
~~~

### 6. 变形操作

##### view/resize/reshape  调整Tensor的形状

- 元素总数必须相同  
- view 和 reshape 可以使用 -1 自动计算维度
- 共享内存

!!!  `view()` 操作是需要 Tensor 在内存中连续的， 这种情况下需要使用 `contiguous()` 操作先将内存变为连续。 对于reshape 操作， 可以看做是 `Tensor.contiguous().view()`.

~~~python
>>> a = torch.Tensor(2,2)
>>> a
tensor([[6.0000e+00, 8.0000e+00],
        [1.0000e+00, 1.8367e-40]])
>>> a.resize(4, 1)
tensor([[6.0000e+00],
        [8.0000e+00],
        [1.0000e+00],
        [1.8367e-40]])
~~~

##### transpose / permute  各维度之间的变换， 

transpose 可以将指定的两个维度的元素进行转置， permute 则可以按照指定的维度进行维度变换

~~~python
>>> x
tensor([[[-0.9699, -0.3375, -0.0178]],
        [[ 1.4260, -0.2305, -0.2883]]])

>>> x.shape
torch.Size([2, 1, 3])
>>> x.transpose(0, 1) # shape => torch.Size([1, 2, 3])
tensor([[[-0.9699, -0.3375, -0.0178],
         [ 1.4260, -0.2305, -0.2883]]])
>>> x.permute(1, 0, 2) # shape => torch.Size([1, 2, 3])
tensor([[[-0.9699, -0.3375, -0.0178],
         [ 1.4260, -0.2305, -0.2883]]])
>>> 
~~~

##### squeeze(dim) / unsquence(dim)  

处理size为1的维度， 前者用于去除size为1的维度， 而后者则是将指定的维度的size变为1

~~~python
>>> a = torch.arange(1, 4)
>>> a
tensor([1, 2, 3]) # shape => torch.Size([3])
>>> a.unsqueeze(0) # shape => torch.Size([1, 3])
>>> a.unqueeze(0).squeeze(0) # shape => torch.Size([3])
~~~

##### expand / expand_as / repeat复制元素来扩展维度

有时需要采用复制的形式来扩展 Tensor 的维度， 这时可以使用 `expand`， `expand()` 函数将 size 为 1的维度复制扩展为指定大小， 也可以用 `expand_as() `函数指定为 示例 Tensor 的维度。

!! `expand` 扩大 tensor 不需要分配新内存，只是仅仅新建一个 tensor 的视图，其中通过将 stride 设为0，一维将会扩展位更高维。

`repeat` 沿着指定的维度重复 tensor。 不同于 `expand()`，复制的是 tensor 中的数据。

~~~python
>>> a = torch.rand(2, 2, 1)
>>> a
tensor([[[0.3094],
         [0.4812]],

        [[0.0950],
         [0.8652]]])
>>> a.expand(2, 2, 3) # 将第2维的维度由1变为3， 则复制该维的元素，并扩展为3
tensor([[[0.3094, 0.3094, 0.3094],
         [0.4812, 0.4812, 0.4812]],

        [[0.0950, 0.0950, 0.0950],
         [0.8652, 0.8652, 0.8652]]])

>>> a.repeat(1, 2, 1) # 将第二位复制一次
tensor([[[0.3094],
         [0.4812],
         [0.3094],
         [0.4812]],

        [[0.0950],
         [0.8652],
         [0.0950],
         [0.8652]]])
~~~

##### 使用切片操作扩展多个维度

~~~
b = a[:,None, None,:] # None 处的维度为１
~~~

### 7. 组合与分块

**组合操作** 是将不同的 Tensor 叠加起来。 主要有 `cat()` 和 `torch.stack()` 两个函数，cat 即 concatenate 的意思， 是指沿着已有的数据的某一维度进行拼接， 操作后的数据的总维数不变， 在进行拼接时， 除了拼接的维度之外， 其他维度必须相同。 而` torch. stack()` 函数会新增一个维度， 并按照指定的维度进行叠加。

~~~shell
torch.cat(list_of_tensors, dim=0)　  # k个(m,n) -> (k*m, n)
torch.stack(list_of_tensors, dim=0)   # k个(m,n) -> (k*m*n)
~~~

**分块操作** 是指将 Tensor 分割成不同的子 Tensor，主要有 `torch.chunk()` 与 `torch.split()` 两个函数，前者需要指定分块的数量，而后者则需要指定每一块的大小，以整形或者list来表示。

~~~python
>>> a = torch.Tensor([[1,2,3], [4,5,6]])
>>> torch.chunk(a, 2, 0)
(tensor([[1., 2., 3.]]), tensor([[4., 5., 6.]]))
>>> torch.chunk(a, 2, 1)
(tensor([[1., 2.],
        [4., 5.]]), tensor([[3.],
        [6.]]))
>>> torch.split(a, 2, 0)
(tensor([[1., 2., 3.],
        [4., 5., 6.]]),)
>>> torch.split(a, [1, 2], 1)
(tensor([[1.],
        [4.]]), tensor([[2., 3.],
        [5., 6.]]))
~~~

### 8. linear algebra

~~~python
trace  # 对角线元素之和(矩阵的迹)
diag  # 对角线元素
triu/tril  # 矩阵的上三角/下三角
addmm/addbmm/addmv/addr/badbmm...  # 矩阵运算
t # 转置
dor/cross # 内积/外积
inverse # 矩阵求逆
svd  # 奇异值分解

torch.mm(tensor1, tensor2)   # 矩阵乘法  (m*n) * (n*p) -> (m*p)
torch.bmm(tensor1, tensor2) # batch的矩阵乘法: (b*m*n) * (b*n*p) -> (b*m*p).
torch.mv(tensor, vec) #　矩阵向量乘法 (m*n) * (n) = (m)
tensor1 * tensor2 # Element-wise multiplication.
~~~

### 9. 基本机制

##### 广播机制

不同形状的 Tensor 进行计算时， 可以自动扩展到较大的相同形状再进行计算。 广播机制的前提是一个 Tensor  至少有一个维度，且从尾部遍历 Tensor 时，两者维度必须相等， 其中七个要么是1， 要么不存在

##### 向量化操作

可以在同一时间进行批量地并行计算，例如矩阵运算，以达到更高的计算效率的一种方式:

##### 共享内存机制

(1) 直接通过 Tensor 来初始化另一个 Tensor， 或者通过 Tensor 的组合、分块、索引、变形来初始化另一个Tensor， 则这两个 Tensor 共享内存:

~~~python
>>> a = torch.randn(2,3)
>>> b = a
>>> c = a.view(6)
>>> b[0, 0] = 0
>>> c[3] = 4
>>> a
tensor([[ 0.0000,  0.3898, -0.7641],
        [ 4.0000,  0.6859, -1.5179]])
~~~

(2) 对于一些操作通过加后缀  “\_”  实现 inplace 操作， 如 `add_()` 和 `resize_()` 等， 这样操作只要被执行， 本身的 Tensor 就会被改变。

~~~
>>> a
tensor([[ 0.0000,  0.3898, -0.7641],
        [ 4.0000,  0.6859, -1.5179]])
>>> a.add_(a)
tensor([[ 0.0000,  0.7796, -1.5283],
        [ 8.0000,  1.3719, -3.0358]])
~~~

(3) Tensor与 Numpy 可以高效的完成转换， 并且转换前后的变量共享内存。在进行 Pytorch 不支持的操作的时候， 甚至可以曲线救国， 将 Tensor 转换为 Numpy 类型，操作后再转化为 Tensor

~~~
# tensor <--> numpy
b = a.numpy() # tensor -> numpy
a = torch.from_numpy(a) # numpy -> tensor
~~~

!!! 需要注意的是，`torch.tensor()` 总是会进行数据拷贝，新 tensor 和原来的数据不再共享内存。所以如果你想共享内存的话，建议使用 `torch.from_numpy()` 或者 `tensor.detach()` 来新建一个 tensor, 二者共享内存。

### 10. nn

~~~python
from torch import nn
import torch.nn.functional as F
~~~

##### pad 填充

~~~python
nn.ConstantPad2d(padding, value)
~~~

##### 卷积和反卷积

~~~python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
~~~

~~~python
#　最常用的两种卷积层设计 3x3 & 1x1
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
~~~

##### 池化层

~~~python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
nn.AdaptiveMaxPool2d(output_size, return_indices=False)
nn.AdaptiveAvgPool2d(output_size)  # global avg pool: output_size=1
nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
~~~

##### 全连接层

~~~python
nn.Linear(in_features, out_features, bias=True)
~~~

##### 防止过拟合相关层

~~~python
nn.Dropout2d(p=0.5, inplace=False)
nn.AlphaDropout(p=0.5)
nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
~~~

##### 激活函数

~~~python
nn.Softplus(beta=1, threshold=20)
nn.Tanh()
nn.ReLU(inplace=False)    
nn.ReLU6(inplace=False)
nn.LeakyReLU(negative_slope=0.01, inplace=False)
nn.PReLU(num_parameters=1, init=0.25)
nn.SELU(inplace=False)
nn.ELU(alpha=1.0, inplace=False)
~~~

##### RNN 

~~~python
nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
nn.RNN(*args, **kwargs)
nn.LSTMCell(input_size, hidden_size, bias=True)
nn.LSTM(*args, **kwargs)
nn.GRUCell(input_size, hidden_size, bias=True)
nn.GRU(*args, **kwargs)
~~~

##### Embedding

~~~python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None)
~~~

##### Sequential

~~~python
nn.Sequential(*args)
~~~

##### loss functon

~~~python
nn.BCELoss(weight=None, size_average=True, reduce=True)
nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
# CrossEntropyLoss 等价于 log_softmax + NLLLoss
nn.L1Loss(size_average=True, reduce=True)
nn.KLDivLoss(size_average=True, reduce=True)
nn.MSELoss(size_average=True, reduce=True)
nn.NLLLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
nn.NLLLoss2d(weight=None, size_average=True, ignore_index=-100, reduce=True)
nn.SmoothL1Loss(size_average=True, reduce=True)
nn.SoftMarginLoss(size_average=True, reduce=True)
nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False, size_average=True, reduce=True)
nn.CosineEmbeddingLoss(margin=0, size_average=True, reduce=True)
~~~

##### functional

~~~python
nn.functional # nn中的大多数layer，在functional中都有一个与之相对应的函数。
              # nn.functional中的函数和nn.Module的主要区别在于，
              # 用nn.Module实现的layers是一个特殊的类，都是由 class layer(nn.Module)定义，
              # 会自动提取可学习的参数。而nn.functional中的函数更像是纯函数，
              # 由def function(input)定义。
~~~

##### init

~~~python
torch.nn.init.uniform
torch.nn.init.normal
torch.nn.init.kaiming_uniform
torch.nn.init.kaiming_normal
torch.nn.init.xavier_normal
torch.nn.init.xavier_uniform
torch.nn.init.sparse
~~~

##### net

~~~python
class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
        self.layer_name = xxxx

    def forward(self, x): 
        x = self.layer_name(x)        
        return x

net.parameters()   # 获取参数 
net.named_parameters  # 获取参数及名称
net.zero_grad()  # 网络所有梯度清零, grad 在反向传播过程中是累加的(accumulated)，
                 # 这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
~~~

### 11. optim -> form torch import optim

~~~python
import torch.optim as optim

optim.SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
optim.Optimizer(params, defaults)

optimizer.zero_grad()  # 等价于 net.zero_grad() 
optimizer.step()
~~~

### 12.  learning rate

~~~python
# Reduce learning rate when validation accuarcy plateau.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
# Cosine annealing learning rate.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
# Reduce learning rate by 10 at given epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
# Learning rate warmup by 10 epochs.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: t / 10)

for t in range(0, 10):
    scheduler.step()
    train(...); val(...)
~~~

### 12. save and load model

~~~python
torch.save(model.state_dict(), 'xxxx_params.pth')
model.load_state_dict(t.load('xxxx_params.pth'))

torch.save(model, 'xxxx.pth')
model.torch.load('xxxx.pth')

all_data = dict(
    optimizer = optimizer.state_dict(),
    model = model.state_dict(),
    info = u'model and optim parameter'
)

t.save(all_data, 'xxx.pth')
all_data = t.load('xxx.pth')
all_data.keys()
~~~

### 13. torchvision

##### models

~~~python
from torchvision import models
resnet34 = models.resnet34(pretrained=True, num_classes=1000)
~~~

##### data augmentation  

~~~python
from torchvision import transforms

# transforms.CenterCrop           transforms.Grayscale           transforms.ColorJitter          
# transforms.Lambda               transforms.Compose             transforms.LinearTransformation 
# transforms.FiveCrop             transforms.Normalize           transforms.functional           
# transforms.Pad                  transforms.RandomAffine        transforms.RandomHorizontalFlip  
# transforms.RandomApply          transforms.RandomOrder         transforms.RandomChoice         
# transforms.RandomResizedCrop    transforms.RandomCrop          transforms.RandomRotation        
# transforms.RandomGrayscale      transforms.RandomSizedCrop     transforms.RandomVerticalFlip   
# transforms.ToTensor             transforms.Resize              transforms.transforms                                           
# transforms.TenCrop              transforms.Scale               transforms.ToPILImage
~~~

##### 自定义 dataset

~~~python
from torch.utils.data import Dataset

class my_data(Dataset):
    def __init__(self, image_path, annotation_path, transform=None):
        # 初始化， 读取数据集
    def __len__(self):
        # 获取数据集的总大小
    def __getitem__(self, id):
        # 对于制定的 id, 读取该数据并返回    
~~~

**datasets**

~~~python
from torch.utils.data import Dataset, Dataloader
from torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.ToTensor(), # convert to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalization

dataset = ImageFolder(root, transform=transform, target_transform=None, loader=default_loader)
dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1,shuffle=True)
for batch_datas, batch_labels in dataloader:
    ...
~~~

##### img process

~~~python
img = make_grid(next(dataiter)[0], 4) 
save_image(img, 'a.png')
~~~

##### data Visualization

~~~python
from torchvision.transforms import ToPILImage

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

(data, label) = trainset[100]
show((data + 1) / 2).resize((100, 100))  # 应该会自动乘以 255 的
~~~

### 14. Code Samples

~~~python
# torch.device object used throughout this script
device = torch.device("cuda" if use_cuda else "cpu")

model = MyRNN().to(device)

# train
total_loss = 0
for input, target in train_loader:
    input, target = input.to(device), target.to(device)
    hidden = input.new_zeros(*h_shape)  # has the same device & dtype as `input`
    ...  # get loss and optimize
    total_loss += loss.item()           # get Python number from 1-element Tensor

# evaluate
with torch.no_grad():                   # operations inside don't track history
    for input, target in test_loader:
        ...
~~~

### 15. jit & torchscript

~~~python
from torch.jit import script, trace
torch.jit.trace(model, torch.rand(1,3,224,224)) 　# export model
@torch.jit.script
~~~

~~~cpp
#include <torch/torch.h>
#include <torch/script.h>

# img blob -> img tensor
torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
img_tensor = img_tensor.permute({0, 3, 1, 2});
img_tensor = img_tensor.toType(torch::kFloat);
img_tensor = img_tensor.div(255);
# load model
std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("resnet.pt");
# forward
torch::Tensor output = module->forward({img_tensor}).toTensor();
~~~

### 16. onnx

~~~python
torch.onnx.export(model, dummy data, xxxx.proto) # exports an ONNX formatted

model = onnx.load("alexnet.proto")               # load an ONNX model
onnx.checker.check_model(model)                  # check that the model

onnx.helper.printable_graph(model.graph)         # print a human readable　representation of the graph
~~~

### 17. Distributed Training

~~~python
import torch.distributed as dist          # distributed communication
from multiprocessing import Process       # memory sharing processes
~~~



## 三. How to Build a network

### 基本工作流程

1. 相关工作调研:  **评价指标、数据集、经典解决方案、待解决问题和已有方案的不同、精度和速度预估、相关难点 ! **

2. 数据探索和方案确定
3. 依次编写模型 models.py、数据集读取接口 datasets.py 、损失函数 losses.py 、评价指标 criterion.py
4. 编写训练脚本(train.py)和测试脚本(test.py)
5. 训练、调试和测评
6. 模型的部署

注意，不要将所有层和模型放在同一个文件中。最佳做法是将最终网络分离为单独的文件（networks.py），并将层、损耗和 ops 保存在各自的文件（layers.py、losses.py、ops.py）中。完成的模型（由一个或多个网络组成）应在一个文件中引用，文件名为 yolov3.py、dcgan.py 这样。

##### (1) 构建神经网络

自定义的网络继承自一般继承自　nn.Module 类，　必须有一个 forward 方法来实现各个层或操作的 forward 传递，　

对于具有**单个输入**和**单个输出**的简单网络，请使用以下模式：

~~~python
class ConvBlock(nn.Module):
  def __init__(self):
    super(ConvBlock, self).__init__()
    self.block = nn.Squential(
       nn.Conv2d(...),
       nn.ReLU(),
       nn.BatchNorm2d(...)
    )
   
  def forward(self, x):
    return self.block(x)

class SimpleNetwork(nn.Module):
    def __init__(self, num_of_layers = 15):
        super(SimpleNetwork, self).__init__()
        layers = list()
        for i in range(num_of_layers):
            layers.append(..)
        self.conv0 = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        return out
~~~

我们建议将网络拆分为更小的**可重用部分**。网络由操作或其它网络模块组成。损失函数也是神经网络的模块，因此可以直接集成到网络中。

##### (2) 自定义数据集

~~~python
class CustomDataset(Dataset):
    """ CustomDataset. """
    def __init__(self, root_dir='./data', transform=None):
        """
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train_data = ...
        self.train_target = ...

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = Image.open(self.train_data[idx])
        target = Image.open(self.train_target[idx])

        if self.transform:
            data, target = self.transform(data, target)

        sample = {'data': data, 'high_img': target}
        return sample
~~~

##### (3) 自定义损失

虽然 PyTorch 已经有很多标准的损失函数，但有时也可能需要创建自己的损失函数。为此，请创建单独的文件 **losses.py** 并扩展**nn.module** 类以创建自定义的损失函数：

~~~python
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        """ CustomLoss"""
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.square(x  - y))
~~~

##### (4) 推荐使用的用于训练模型的代码结构

~~~python
# import statements
import torch
import torch.nn as nn
from torch.utils import data
...

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
...
  
# dataset
transform_train = ...
trainform_text = ...

train_dataset = CustomDataset(args.train_dataset, is_trainval = True, transform = transform_train) 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0, drop_last=False) 
valid_dataset = CustomDataset(args.valid_dataset, is_trainval = True, transform = transform_test)  
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.val_batch_size, 
                                           shuffle=True, num_workers=0) 
# model & loss
net = CustomNet().to(device) 
criterion = ...  
# lr & optimizer
optim = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)


# load resume
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

def train(epoch):
    model.train()　# 在　model(x)　前需要添加　model.eval()　或者　model.eval()

    avg_loss = 0.0
    train_acc = 0.0
    for batch_idx, batchdata in enumerate(train_loader):
        data, target = batchdata["data"], batchdata["target"] #
        data, target = data.to(device), target.to(device)  #
        # 在 loss.backward()　前用　optimizer.zero_grad()　清除累积梯度
        optimizer.zero_grad() # optimizer.zero_grad　与　model.zero_grad效果一样

        predict = model(data) # 
        loss = criterion(predict, target) #
        avg_loss += loss.item() #

        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    if (epoch + 1) %  args.save_interval == 0:
        state = { 'epoch': epoch + 1,
                   'state_dict': model.state_dict(),
                   'best_prec': 0.0,
                   'optimizer': optimizer.state_dict()}
        model_path = os.path.join(args.checkpoint_dir, 'model_' + str(epoch) + '.pth')
        torch.save(state, model_path)


def test():
    model.eval()

    test_loss = 0
    for batch_idx, batchdata in enumerate(valid_loader):
        data, target = batchdata["data"], batchdata["target"] #
        data, target = data.to(device), target.to(device) #
        predict = model(data) # 
        test_loss += criterion(predict, target) #
        psnr = criterion(predict * 255, target * 255) #

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, loss:{}, PSNR: ({:.1f})\n'.format(
        test_loss, test_loss / len(valid_loader.dataset), psnr / len(valid_loader.dataset)))
    return psnr / float(len(valid_loader.dataset))


best_prec = 0.0
for epoch in range(args.start_epoch, args.epochs):
    train(epoch)
    scheduler.step()
    print(print(optimizer.state_dict()['param_groups'][0]['lr']))

    current_prec = test() 
    is_best = current_prec > best_prec #　更改大小写 !
    best_prec = max(best_prec, best_prec) #  max or min

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec': best_prec,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.checkpoint_dir)
~~~



## 四. 常见代码片段

### 1. 基础配置

##### (1) check pytorch version

~~~python
torch.__version__               # PyTorch version
torch.version.cuda              # Corresponding CUDA version
torch.backends.cudnn.version()  # Corresponding cuDNN version
torch.cuda.get_device_name(0)   # GPU type
~~~

##### (2) update pytorch

~~~
conda update pytorch torchvision -c pytorch
~~~

##### (3) random seed setting 

~~~
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
~~~

##### (4) 指定程序运行在特定显卡上：

在命令行指定环境变量

```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

在代码中指定

~~~
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
~~~

##### (5) 判断是否有CUDA支持

```
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')   
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

##### (6) 设置为cuDNN benchmark模式

Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。

~~~
toch.backends.cudnn.benchmark = True
~~~

如果想要避免这种结果波动，设置

~~~
torch.backends.cudnn.deterministic = True
~~~

##### (7) 手动清除GPU存储

有时Control-C中止运行后GPU存储没有及时释放，需要手动清空。在PyTorch内部可以

~~~
torch.cuda.empty_cache() 
~~~

或在命令行可以先使用ps找到程序的PID，再使用kill结束该进程

~~~
 ps aux | grep python    kill -9 [pid] 
~~~

或者直接重置没有被清空的GPU

~~~
nvidia-smi --gpu-reset -i [gpu_id]
~~~

### 2. 模型

##### (1) 提取ImageNet预训练模型某层的卷积特征

~~~
# VGG-16 relu5-3 feature.
model = torchvision.models.vgg16(pretrained=True).features
# VGG-16 pool5 feature.
model = torchvision.models.vgg16(pretrained=True)
model = torch.nn.Sequential(model.features, model.avgpool)
# VGG-16 fc7 feature.
model = torchvision.models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
# ResNet GAP feature.
model = torchvision.models.resnet18(pretrained=True)
model = torch.nn.Sequential(collections.OrderedDict(
    list(model.named_children())[:-1]))

with torch.no_grad():
    model.eval()
    conv_representation = model(image)
~~~

##### (2) 提取ImageNet预训练模型多层的卷积特征

~~~
class FeatureExtractor(torch.nn.Module):
    """Helper class to extract several convolution features from the given
    pre-trained model.

    Attributes:
        _model, torch.nn.Module.
        _layers_to_extract, list<str> or set<str>

    Example:
        >>> model = torchvision.models.resnet152(pretrained=True)
        >>> model = torch.nn.Sequential(collections.OrderedDict(
                list(model.named_children())[:-1]))
        >>> conv_representation = FeatureExtractor(
                pretrained_model=model,
                layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'})(image)
    """
    def __init__(self, pretrained_model, layers_to_extract):
        torch.nn.Module.__init__(self)
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)
    
    def forward(self, x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
            return conv_representation
~~~

##### (３)  部分层使用预训练模型

注意如果保存的模型是`torch.nn.DataParallel`，则当前的模型也需要是`torch.nn.DataParallel`。`torch.nn.DataParallel(model).module == model`。

~~~
   model.load_state_dict(torch.load('model,pth'), strict=False)
~~~

将在GPU保存的模型加载到CPU:

~~~
   model.load_state_dict(torch.load('model,pth', map_location='cpu'))
~~~

##### （４）fine-tune 微调全连接层

##### (4) 微调全连接层

~~~
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 100)  # Replace the last fc layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
~~~

以较大学习率微调全连接层，较小学习率微调卷积层

~~~
model = torchvision.models.resnet18(pretrained=True)
finetuned_parameters = list(map(id, model.fc.parameters()))
conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
parameters = [{'parameters': conv_parameters, 'lr': 1e-3}, 
              {'parameters': model.fc.parameters()}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
~~~

##### （５）保存与加载断点

注意为了能够恢复训练，我们需要同时保存模型和优化器的状态，以及当前的训练轮数。

~~~
# Save checkpoint.
is_best = current_acc > best_acc
best_acc = max(best_acc, current_acc)
checkpoint = {
    'best_acc': best_acc,    
    'epoch': t + 1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
model_path = os.path.join('model', 'checkpoint.pth.tar')
torch.save(checkpoint, model_path)
if is_best:
    shutil.copy('checkpoint.pth.tar', model_path)

# Load checkpoint.
if resume:
    model_path = os.path.join('model', 'checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Load checkpoint at epoch %d.' % start_epoch)
~~~

##### (６) 计算模型参数量[D]

~~~
# Total parameters                    
num_params = sum(p.numel() for p in model.parameters()) 
# Trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
~~~

##### (７) 模型权值初始化[D]

注意`model.modules()`和`model.children()`的区别：`model.modules()`会迭代地遍历模型的所有子层，而`model.children()`只会遍历模型下的一层。

~~~python
# Common practise for initialization.
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                      nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, val=0.0)
    
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)
  
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

# Initialization with given tensor.
m.weight = torch.nn.Parameter(tensor)
~~~

##### (8) 冻结参数

~~~
if not requires_grad:
    for param in self.parameters():
        param.requires_grad = False
~~~

### 3. 数据

##### (1) 常见训练和验证数据预处理

ToTensor操作会将PIL.Image或形状为H×W×D，数值范围为[0, 255]的np.ndarray转换为形状为D×H×W，数值范围为[0.0, 1.0]的torch.Tensor。

~~~
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=224,
                                             scale=(0.08, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
 ])
 val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
])
~~~

### 4. 训练

##### (1) 将整数标记转换成独热（one-hot）编码  

 (PyTorch中的标记默认从0开始)

~~~
   N = tensor.size(0)
   one_hot = torch.zeros(N, num_classes).long()
   one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
~~~

##### (2) 计算两组数据之间的两两欧式距离

~~~
# X1 is of shape m*d.
X1 = torch.unsqueeze(X1, dim=1).expand(m, n, d)
# X2 is of shape n*d.
X2 = torch.unsqueeze(X2, dim=0).expand(m, n, d)
# dist is of shape m*n, where dist[i][j] = sqrt(|X1[i, :] - X[j, :]|^2)
dist = torch.sqrt(torch.sum((X1 - X2) ** 2, dim=2))
~~~

##### (3) 双线性汇合（bilinear pooling）

~~~
X = torch.reshape(N, D, H * W)                        # Assume X has shape N*D*H*W
X = torch.bmm(X, torch.transpose(X, 1, 2)) / (H * W)  # Bilinear pooling
assert X.size() == (N, D, D)
X = torch.reshape(X, (N, D * D))
X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)   # Signed-sqrt normalization
X = torch.nn.functional.normalize(X)                  # L2 normalization
~~~

##### (4) L1 正则化

~~~
l1_regularization = torch.nn.L1Loss(reduction='sum')
loss = ...  # Standard cross-entropy loss
for param in model.parameters():
    loss += torch.sum(torch.abs(param))
loss.backward()


reg = 1e-6
l2_loss = Variable(torch.FloatTensor(1), requires_grad=True)
for name, param in model.named_parameters():
    if 'bias' not in name:
        l2_loss = l2_loss + (0.5 * reg * torch.sum(torch.pow(W, 2)))
~~~

##### (5) 不对偏置项进行L2正则化/权值衰减（weight decay）

~~~
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},                
              {'parameters': others_list}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
~~~

##### (6) 梯度裁剪（gradient clipping）

 ~~~
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
 ~~~

##### (7) 计算Softmax 输出的正确率

~~~
score = model(images)
prediction = torch.argmax(score, dim=1)
num_correct = torch.sum(prediction == labels).item()
accuruacy = num_correct / labels.size(0)
~~~

##### (8) 获取当前学习率

~~~
# If there is one global learning rate (which is the common case).
lr = next(iter(optimizer.param_groups))['lr']
# If there are multiple learning rates for different layers.
all_lr = []
for param_group in optimizer.param_groups:
    all_lr.append(param_group['lr'])
~~~

### 5. Trick

##### (1)  label smothing

~~~
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    N = labels.size(0)
    # C is the number of classes.
    smoothed_labels = torch.full(size=(N, C), fill_value=0.1 / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=0.9)

    score = model(images)
    log_prob = torch.nn.functional.log_softmax(score, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

##### (2) Mixup

~~~
beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # Mixup images.
    lambda_ = beta_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()
    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]

    # Mixup loss.    
    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores, labels) 
            + (1 - lambda_) * loss_function(scores, labels[index]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

##### (3) 多卡同步BN（Batch normalization）

当使用torch.nn.DataParallel将代码运行在多张GPU卡上时，PyTorch的BN层默认操作是各卡上数据独立地计算均值和标准差，同步BN使用所有卡上的数据一起计算BN层的均值和标准差，缓解了当批量大小（batch size）比较小时对均值和标准差估计不准的情况，是在目标检测等任务中一个有效的提升性能的技巧。

参见： [Synchronized-BatchNorm-PyTorchgithub](vacancy/Synchronized-BatchNorm-PyTorchgithub.com)



## 五. 网络优化和加速 [alpha]

### 1. 数据

##### (0) dataloader

- num_workers与batch_size调到合适值，并非越大越快（注意后者也影响模型性能）(需要在实验中找到最快的取值)
- eval/test时shuffle=False
- 内存够大的情况下，dataloader的**pin_memory**设为True。对特别小的数据集如 MNIST 设置 `pin_memory=False`  反而更快一些。

#####  (1) 预处理提速

- 尽量减少每次读取数据时的预处理操作，可以考虑把一些固定的操作，例如 resize ，事先处理好保存下来，训练的时候直接拿来用
- Linux上将预处理搬到GPU上加速：

- - **NVIDIA/DALI** ：https://github.com/NVIDIA/DALI
  - https://github.com/tanglang96/DataLoaders_DALI

- 数据预取：prefetch_generator（[方法](https://zhuanlan.zhihu.com/p/80695364)）让读数据的worker能在运算时预读数据，而默认是数据清空时才读

##### (2) IO 提速

- 使用更快的图片处理：

- - **opencv 一般要比 PIL 要快**
  - 对于jpeg读取，可以尝试 **jpeg4py**
  - 存 **bmp** 图（降低解码时间）

- **小图拼起来存放（降低读取次数）：对于大规模的小文件读取，建议转成单独的文件，可以选择的格式可以考虑**：TFRecord（Tensorflow）、recordIO(recordIO)、hdf5、 pth、n5、lmdb 等等（https://github.com/Lyken17/Efficient-PyTorch#data-loader）

- - **TFRecord**：https://github.com/vahidk/tfrecord
  - 借助 **lmdb 数据库格式**：

- - - https://github.com/Fangyh09/Image2LMDB
    - https://blog.csdn.net/P_LarT/article/details/103208405
    - https://github.com/lartpang/PySODToolBox/blob/master/ForBigDataset/ImageFolder2LMDB.py
    - https://github.com/Lyken17/Efficient-PyTorch

##### (3)　借助硬件

- 借助内存：**直接载到内存里面，或者把把内存映射成磁盘好了**
- 借助固态：把读取速度慢的机械硬盘换成 **NVME 固态**吧～

##### (4) 训练策略

- 在训练中使用**低精度（FP16 甚至 INT8 、二值网络、三值网络）表示取代原有精度（FP32）表示**

- - NVIDIA/Apex：

- - - https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729
    - https://github.com/nvidia/apex

- 使用分布式训练　DDP 或者 horovod

##### (5) 代码层面

- `torch.backends.cudnn.benchmark = True`
- Do numpy-like operations on the GPU wherever you can
- Free up memory using` del`     用`del`及时删除不用的中间变量，节约GPU存储。
- Avoid unnecessary transfer of data from the GPU
- Use pinned memory, and use non_blocking=False to parallelize data transfer and GPU number crunching

### 2. model

1. 用**float16**代替默认的float32运算（[方法参考](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers/blob/dad3c7a485b7ffc6fd2766f349e6ee845ecc2eee/examples/run_classifier.py)，搜索"fp16"可以看到需要修改之处，包括model、optimizer、backward、learning rate）

2. **优化器**以及对应参数的选择，如learning rate，不过它对性能的影响似乎更重要【占坑】

3. 少用循环，多用**向量化**操作

4. 经典操作尽量用别人优化好的**库**，别自己写

5. 数据很多时少用append，虽然使用很方便，不过它每次都会重新分配空间？所以数据很大的话，光一次append就要几秒（测过），可以先分配好整个容器大小，每次用索引去修改内容，这样一步只要0.0x秒

6. 固定对模型影响不大的部分参数，还能节约显存，可以用 detach() 切断反向传播，注意若仅仅给变量设置 required_grad=False 还是会计算梯度的

7. eval/test 的时候，加上 model.eval() 和 torch.no_grad()，前者固定 batch-normalization 和 dropout 但是会影响性能，后者关闭 autograd

8. 提高程序**并行度**，例如 我想 train 时对每个 epoch 都能 test 一下以追踪模型性能变化，但是 test 时间成本太高要一个小时，所以写了个 socket，设一个127.0.0.1 的端口，每次 train 完一个 epoch 就发个UDP过去，那个进程就可以自己 test，同时原进程可以继续 train 下一个 epoch（对 这是自己想的诡异方法hhh）

9. torch.backends.cudnn.benchmark设为True，可以让cudnn根据当前训练各项config寻找优化算法，但这本身需要时间，所以input size在训练时会频繁变化的话，建议设为False

10. 使用`inplace`操作可节约 GPU 存储，如

    ~~~
    x = torch.nn.functional.relu(x, inplace=True)
    ~~~

11. 减少CPU和GPU之间的数据传输。例如， 如果你想知道一个 epoch 中每个 mini-batch 的 loss 和准确率，先将它们累积在 GPU 中等一个 epoch 结束之后一起传输回 CPU 会比每个 mini-batch 都进行一次 GPU 到 CPU 的传输更快。

12. 使用半精度浮点数`half()`会有一定的速度提升，具体效率依赖于GPU型号。需要小心数值精度过低带来的稳定性问题。时常使用 `assert tensor.size() == (N, D, H, W)`作为调试手段，确保张量维度和你设想中一致。

13. 除了标记 y 外，尽量少使用一维张量，使用n*1的二维张量代替，可以避免一些意想不到的一维张量计算结果。

14.  统计代码各部分耗时

~~~python
with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as profile:
    ...
    print(profile)
~~~

或者在命令行运行：

~~~
python -m torch.utils.bottleneck main.py
~~~



## 六. 分布式训练 [alpha]

~~~
os.environ['NCCL_SOCKET_IFNAME'] = 'enp2s0'
os.environ['GLOO_SOCKET_IFNAME'] = 'enp2s0'
~~~

#### nn.DataParallel

~~~
gpus = [0, 1, 2, 3]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
~~~

#### torch.distributed

~~~
parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, default='nccl', help='Name of the backend to use.')
parser.add_argument('-i',
                    '--init-method',
                    type=str,
                    default='env://',
                    help='URL specifying how to initialize the package.')
parser.add_argument('-ws', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
args = parser.parse_args()
~~~

~~~
distributed.init_process_group(
    backend=args.backend,
    init_method=args.init_method,
    world_size=args.world_size,
    rank=args.rank,
)
~~~

~~~
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
~~~

~~~
model = nn.parallel.DistributedDataParallel(model)
~~~

####　torch.multiprocessing

~~~
import torch.multiprocessing as mp
mp.spawn(main_worker, nprocs=4, args=(4, myargs))
~~~

#### APEX

~~~
from apex import amp
from apex.parallel import DistributedDataParallel
~~~

~~~
model, optimizer = amp.initialize(model, optimizer)
model = DistributedDataParallel(model)

with amp.scale_loss(loss, optimizer) as scaled_loss:
   scaled_loss.backward()
~~~

#### Horovod

```
import horovod.torch as hvd

hvd.local_rank()
```

~~~
hvd.init()
~~~

~~~
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
~~~

~~~
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
~~~

~~~
hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.fp16)
~~~



## 七. 移动端部署

- 模型压缩和加速(量化、剪枝)
- Pytorch to onnx to X



## 八. 服务器端部署 

- flask_api
- TensorRT



## 九. 最佳实践(To do or not to do)

**在「nn.Module」的「forward」方法中避免使用 Numpy 代码**

Numpy 是在 CPU 上运行的，它比 torch 的代码运行得要慢一些。由于 torch 的开发思路与 numpy 相似，所以大多数  中的函数已经在 PyTorch 中得到了支持。



## 十. ToolBox [alpha]

### 1. 预训练模型

https://github.com/Cadene/pretrained-models.pytorch

https://github.com/rwightman/pytorch-image-models

https://github.com/welkin-feng/ComputerVision

### 2. 数据增强

https://github.com/albumentations-team/albumentations

### 3. 标记工具

[**Labelme:**](https://github.com/wkentaro/labelme) Image Polygonal Annotation with Python

[**LabelImg**](https://github.com/tzutalin/labelImg)：LabelImg is a graphical image annotation tool and label object bounding boxes in images

### 4. 数据集查找

论文的评测指标　=> datasets

[**Kaggle**](https://www.kaggle.com/)

[**Google Datasets Search Engine**](https://toolbox.google.com/datasetsearch)

[**Microsoft Datasets**](https://msropendata.com/)

[**Computer Vision Datasets**](https://www.visualdata.io/)

[**Github awesomedata**](https://github.com/awesomedata/awesome-public-datasets)

[**UCI Machine Learning Repository.**](https://archive.ics.uci.edu/ml/datasets.html)

[**Amazon Datasets**](https://registry.opendata.aws/)

**Government Datasets:** [**EU**](https://data.europa.eu/euodp/data/dataset) [**US**](https://www.data.gov/) [**NZL**](https://catalogue.data.govt.nz/dataset) [**IND**](https://data.gov.in/)

### 5. 模型分析工具

##### (1) 卷积层输出大小计算

##### https://ezyang.github.io/convolution-visualizer/index.html

##### (2) 计算模型参数量

https://github.com/sksq96/pytorch-summary

##### (3) 模型可视化工具

[**Netron:**](https://github.com/lutzroeder/Netron) now supports **ONNX**, **Keras**, **CoreML**, **Caffe2**, **Mxnet**, **Pytorch** and **Tensorflow**.

[**Graphviz:**](https://github.com/szagoruyko/pytorchviz) **Pytorch**

### 6. 可视化工具

[visdom](https://github.com/facebookresearch/visdom)

~~~python
# Example using Visdom.
vis = visdom.Visdom(env='Learning curve', use_incoming_socket=False)
assert self._visdom.check_connection()
self._visdom.close()
options = collections.namedtuple('Options', ['loss', 'acc', 'lr'])(
    loss={'xlabel': 'Epoch', 'ylabel': 'Loss', 'showlegend': True},
    acc={'xlabel': 'Epoch', 'ylabel': 'Accuracy', 'showlegend': True},
    lr={'xlabel': 'Epoch', 'ylabel': 'Learning rate', 'showlegend': True})

for t in epoch(80):
    tran(...)
    val(...)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([train_loss]),
             name='train', win='Loss', update='append', opts=options.loss)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([val_loss]),
             name='val', win='Loss', update='append', opts=options.loss)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([train_acc]),
             name='train', win='Accuracy', update='append', opts=options.acc)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([val_acc]),
             name='val', win='Accuracy', update='append', opts=options.acc)
    vis.line(X=torch.Tensor([t + 1]), Y=torch.Tensor([lr]),
             win='Learning rate', update='append', opts=options.lr)
~~~

[Tensorboard](https://pytorch.org/docs/stable/tensorboard.html)

- **acc / loss**

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter()
for n_iter in range(100):
    dummy_s1 = torch.rand(1)
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
writer.close()
```

- **img**

```python
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
writer = SummaryWriter()
if n_iter % 10 == 0:
    x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
    writer.add_image('Image', x, n_iter)
writer.close()
```

- 在一张图中加入两条曲线

```python
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
```

### 7. Pytorch 加速

**NVIDIA/DLAI:** https://github.com/NVIDIA/DALI

**Efficient-Pytorch:** https://github.com/Lyken17/Efficient-PyTorch

**NVIDIA/APEX:** https://github.com/nvidia/apex

### 8. 性能分析工具

- nvidia-smi
- htop
- iotop
- nvtop
- py-spy
- strace



## 参考链接 [alpha]

- Tensorflow cookbook

- https://github.com/kevinzakka/pytorch-goodies

- https://github.com/chenyuntc/pytorch-book

- pytorch 官方文档和tutorial

- https://github.com/IgorSusmelj/pytorch-styleguide


