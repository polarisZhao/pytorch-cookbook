# Pytorch API

### 1. import torch

import & vision

~~~python
import torch 
print(torch.__version__)
~~~

### 2. Tensor type 🌟

Pytorch 给出了 9 种 CPU Tensor 类型和 9 种 GPU Tensor 类型。Pytorch 中默认的数据类型是torch.FloatTensor, 即 torch.Tensor 等同于 torch.FloatTensor。

| Data type                | dtype                         | CPU tensor                            | GPU tensor              |
| ------------------------ | ----------------------------- | ------------------------------------- | ----------------------- |
| 32-bit floating point    | torch.float32 or torch.float  | torch.FloatTensor                     | torch.cuda.FloatTensor  |
| 64-bit floating point    | torch.float64 or torch.double | torch.DoubleTensor                    | torch.cuda.DoubleTensor |
| 16-bit floating point    | torch.float16 or torch.half   | torch.HalfTensor                      | torch.cuda.HalfTensor   |
| 8-bit integer (unsigned) | torch.uint8                   | torch.ByteTensor                      | torch.cuda.ByteTensor   |
| 8-bit integer (signed)   | torch.int8                    | torch.CharTensor                      | torch.cuda.CharTensor   |
| 16-bit integer (signed)  | torch.int16 or torch.short    | torch.ShortTensor                     | torch.cuda.ShortTensor  |
| 32-bit integer (signed)  | torch.int32 or torch.int      | torch.IntTensor                       | torch.cuda.IntTensor    |
| 64-bit integer (signed)  | torch.int64 or torch.long     | torch.LongTensor                      | torch.cuda.LongTensor   |
| Boolean                  | torch.bool                    | [torch.BoolTensor](#torch.BoolTensor) | torch.cuda.BoolTensor   |

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
>>> a.type(torch.DoubleTensor)   # 使用 dtype() 函数进行转换
tensor([[0.6065, 0.0122, 0.4473],
        [0.5937, 0.5530, 0.4663]], dtype=torch.float64)
>>> a = a.double()  # 直接使用 int()、float() 和 double() 等直接进行数据类型转换进行
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

##### Tensor 相关信息获取

~~~python
torch.size()/torch.shape   # 两者等价， 返回t的形状, 可以使用 x.size()[1] 或 x.size(1) 查看列数
torch.numel() / torch.nelement()  # 两者等价, t中元素总个数
a.item()  # 取出单个tensor的值
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
ones(sizes)  # 全 1 Tensor     
zeros(sizes)  # 全 0 Tensor
eye(sizes)  # 对角线为1，不要求行列一致
full(sizes, value) # 指定 value
~~~

##### 分布

~~~python
rand(sizes)  # 均匀分布   
randn(sizes)   # 标准分布

# 正态分布: 返回一个张量，包含从给定参数 means,std 的离散正态分布中抽取随机数。 均值 means是一个张量，包含每个输出元素相关的正态分布的均值 -> 以此张量的均值作为均值
# std是一个张量，包含每个输出元素相关的正态分布的标准差 -> 以此张量的标准差作为标准差。 均值和标准差的形状不须匹配，但每个张量的元素个数须相同
torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([-0.1987,  3.1957,  3.5459,  2.8150,  5.5398,  5.6116,  7.5512,  7.8650,
         9.3151, 10.1827])
uniform(from,to) # 均匀分布 

arange(s, e, steps)  # 从s到e，步长为step
linspace(s, e, num)   # 从s到e,均匀切分为 num 份, 注意linespace和arange的区别，前者的最后一个参数是生成的Tensor中元素的数量，而后者的最后一个参数是步长。
randperm(m) # 0 到 m-1 的随机序列
~~~

### 4. 索引、比较、排序

##### 索引操作

~~~python
a[row, column]  # row 行， cloumn 列
a[index]  # 第index 行
a[:,index]  # 第 index 列

a[0, -1] #第零行， 最后一个元素
a[:index] # 前 index 行
a[:row, 0:1] # 前 row 行， 0和1列

a[a>1] # 选择 a > 1的元素， 等价于 a.masked_select(a>1)
torch.nonzero(a) # 选择非零元素的坐标，并返回
a.clamp(x, y) # 对 Tensor 元素进行限制， 小于x用x代替， 大于y用y代替
torch.where(condition, x, y) # 满足condition 的位置输出x， 否则输出y
>>> a
tensor([[ 6., -2.],
        [ 8.,  0.]])
>>> torch.where(a>1, torch.full_like(a, 1), a) # 大于1 的部分直接用1代替， 其他保留原值
tensor([[ 1., -2.],
        [ 1.,  0.]])
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

Element-wise: 输出的 tensor 形状与原始的形状一致

~~~python
abs/sqrt/div/exp/fmod/log/pow...
cos/sin/asin/atan2/cosh...
ceil/round/floor/trunc
clamp(input, min, max)
sigmoid/tanh...
~~~

归并操作： 输出的tensor形状小于原始的 Tensor形状

~~~python
mean/sum/median/mode   # 均值/和/ 中位数/众数
norm/dist  # 范数/距离
std/var  # 标准差/方差
cumsum/cumprd # 累加/累乘
~~~



### 6. 变形操作

##### view/resize/reshape **调整Tensor的形状**

- 元素总数必须相同  
- view 和 reshape 可以使用-1自动计算维度
- 共享内存

! view() 操作是需要 Tensor 在内存中连续的， 这种情况下需要使用 contiguous() 操作先将内存变为连续。 对于reshape 操作， 可以看做是 `Tensor.contiguous().view()`.

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

##### squeeze(dim)/unsquence(dim)  

处理size为1的维度， 前者用于去除size为1的维度， 而后者则是将指定的维度的size变为1

~~~python
>>> a = torch.arange(1, 4)
>>> a
tensor([1, 2, 3]) # shape => torch.Size([3])
>>> a.unsqueeze(0) # shape => torch.Size([1, 3])
>>> a.unqueeze(0).squeeze(0) # shape => torch.Size([3])
~~~

##### expand/ expand_as/repeat复制元素来扩展维度

有时需要采用复制的形式来扩展 Tensor 的维度， 这时可以使用 expand， expand() 函数将 size 为 1的维度复制扩展为指定大小， 也可以用 expand_as() 函数指定为 示例 Tensor 的维度。

!! expand 扩大 tensor 不需要分配新内存，只是仅仅新建一个 tensor 的视图，其中通过将 stride 设为0，一维将会扩展位更高维。

repeat 沿着指定的维度重复 tensor。 不同于 expand()，复制的是 tensor 中的数据。

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



### 7. 组合与分块

**组合操作** 是将不同的 Tensor 叠加起来。 主要有 `cat()` 和 `torch.stack()` 两个函数，cat 即 concatenate 的意思， 是指沿着已有的数据的某一维度进行拼接， 操作后的数据的总维数不变， 在进行拼接时， 除了拼接的维度之外， 其他维度必须相同。 而` torch. stack()` 函数 指新增维度， 并按照指定的维度进行叠加。

**分块操作** 是指将 Tensor 分割成不同的子Tensor，主要有 `torch.chunk()` 与 `torch.split()` 两个函数，前者需要指定分块的数量，而后者则需要指定每一块的大小，以整形或者list来表示。

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
mm/bmm   # 矩阵的乘法， batch的矩阵乘法
addmm/addbmm/addmv/addr/badbmm...  # 矩阵运算
t # 转置
dor/cross # 内积/外积
inverse # 矩阵求逆
svd  # 奇异值分解
~~~



### 9. 基本机制

##### 广播机制

不同形状的Tensor 进行计算时， 可以自动扩展到较大的相同形状再进行计算。 广播机制的前提是一个Tensor 至少有一个维度，且从尾部遍历Tensor时，两者维度必须相等， 其中七个要么是1， 要么不存在

##### 向量化操作

可以在同一时间进行批量地并行计算，例如矩阵运算，以达到更高的计算效率的一种方式:

##### 共享内存机制

(1) 直接通过 Tensor 来初始化另一个 Tensor， 或者通过 Tensor 的组合、分块、索引、变形来初始化另一个Tensor， 则这两个Tensor 共享内存:

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

(2) 对于一些操作通过加前缀  “\_”  实现 inplace 操作， 如 `add_()` 和 `resize_()` 等， 这样操作只要被执行， 本身的 Tensor 就会被改变。

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

!! 需要注意的是，**torch.tensor() 总是会进行数据拷贝，新tensor和原来的数据不再共享内存**。所以如果你想共享内存的话，建议使用 `torch.from_numpy()` 或者 `tensor.detach()` 来新建一个tensor, 二者共享内存。



### 10. nn

~~~python
from torch import nn
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

##### 池化层

~~~python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
nn.AdaptiveMaxPool2d(output_size, return_indices=False)
nn.AdaptiveAvgPool2d(output_size)
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
net.zero_grad()  # 网络所有梯度清零, grad 在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
~~~



### 11. optim -> form torch import optim

~~~python
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

##### datasets

**datasets**

~~~python
transform = transforms.Compose([
        transforms.ToTensor(), # vonvert to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # normalization
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
show = ToPILImage()     # 可以把Tensor转成Image，方便可视化

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

