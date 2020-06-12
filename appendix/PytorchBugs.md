### 一. CUDA & cudnn

**1. cuDNN error:CUDNN_STATUS_EXECUTION_FAILED**

**A：**This happens also in the windows port of PyTorch, the only way to overcome this when using (in my case) large CNN’s is to use: 

~~~shell
torch.backends.cudnn.enabled=False
~~~

**2. out of memory at /opt/conda/conda-bld/pytorch_1524590031827/work/aten/src/THC/generic/THCStorage.cu:58**

**A：**! 显存不够, 没啥办法　-> **Batchsize 改小、加显卡、混合精度训练**

**3. CUDA 设置指定 GPU 可见**

**> 可设置环境变量 CUDA_VISIBLE_DEVICES，指明可见的 cuda 设备**

**方法1**: 在  `/etc/profile` 或 `~/.bashrc` 的配置文件中配置环境变量(`/etc/profile`影响所有用户，`~/.bashrc `影响当前用户使用的 bash shell)

在 `/etc/profile` 文件末尾添加以下行：

~~~
export CUDA_VISIBLE_DEVICES=0,1 # 仅显卡设备0,1GPU可见
~~~

`:wq` 保存并退出, 然后执行如下命令:

~~~
source /etc/profile
~~~

使配置文件生效

**方法2**：若上述配置无效，可在执行 cuda 程序时指明参数，如

~~~shell
CUDA_VISIBLE_DEVICES=0,1 ./cuda_executable
# Environment Variable Syntax                Results
# CUDA_VISIBLE_DEVICES=1        Only device 1 will be seen
# CUDA_VISIBLE_DEVICES=0,1       Devices 0 and 1 will be visible
# CUDA_VISIBLE_DEVICES="0,1"      Same as above, quotation marks are optional
# CUDA_VISIBLE_DEVICES=0,2,3      Devices 0, 2, 3 will be visible; device 1 is masked
~~~

**4. nn.Module.cuda() 和 Tensor.cuda() 的作用效果差异**

无论是对于模型还是数据，`cuda()` 函数都能实现从 CPU 到 GPU 的内存迁移，但是他们的作用效果有所不同。

对于 `nn.Module`:

~~~python
model = model.cuda() # 等价于 model.cuda() 
~~~

上面两句能够达到一样的效果，即对 model 自身进行的内存迁移。

对于 `Tensor`:

​    和nn.Module不同，调用 `tensor.cuda()` 只是返回这个 tensor 对象在 GPU 内存上的拷贝，而不会对自身进行改变**。因此必须对tensor进行重新赋值，即 **`tensor = tensor.cuda()`.

~~~python
model = create_a_model() 
tensor = torch.zeros([2,3,10,10]) 
model.cuda() 
tensor.cuda() 
model(tensor)    # 会报错 
tensor = tensor.cuda() 
model(tensor)    # 正常运行 
~~~

**5. an illegal memory access was encountered at /opt/conda/conda-bld/pytorch_1525909934016/work/aten/src/THC/generated/../THCReduceAll.cuh:339**

在 GPU 训练中不正确的内存访问，有可能是程序问题也有可能是当前驱动不兼容的问题：

​        因为 cuda 运行是异步的，所以我们的错误信息可能没有那么准确，为此我们将环境变量 `CUDA_LAUNCH_BLOCKING=1` 设为1,在当前的 terminal 中执行 `CUDA_LAUNCH_BLOCKING=1 python3 train.py`  —— (train.py是你要执行的.py文件)，再次执行就可以查看到当前出错的代码行。

​        仔细检查当前的代码，查看是否有内存的不正确访问，最常见的是索引超出范围。 如果不是代码问题，那么有可能是当前的 pytorch 版本和你的显卡型号不兼容，或者cudnn的库不兼容的问题。可以挑选出错误代码段对其进行简单的测试观察有没有错误即可。

**6.** **AttributeError: module 'torch._C' has no attribute '_cuda_getDevice'**

**A:** According to [docs](http://pytorch.org/docs/master/torch.html#torch.load), I believe you should be loading your models with

~~~python
torch.load(file, map_location=device)
~~~



### 二. mismacth 

**1.** **Input type (CUDAFloatTensor) and weight type (CPUFloatTensor) should be the same**

**A:** **输入数据和模型中的权重设备不一样**，**模型的参数不在GPU中，而输入数据在CPU中**，可以通过添加　model.cuda()　将模型转移到GPU上以解决这个问题。

**2. Input type (CUDADoubleTensor) and weight type (CUDAFloatTensor) should be the same**

**A:** **输入数据和模型中的权重数据类型不一致，一个为 Double 一个为 Float**, 通过对输入数据 Tensor(x) 进行 x.float() 将输入数据和模型权重类型一致，或者将模型权重的类型转化为 Double 也可以解决问题。

**3. RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 1 and 3 in dimension 1**

(1) 读取数据的时候发生错误了。一般来说是维度不匹配，如果一个数据集中有 3 通道的也有四通道的图像，总之就是从 dataset 中传入 dataloader 中的图像大小不一致。自己好好检查检查，是否将所有图像都变成同样的 shape 。注意，**只要是 dataset 中的数据都要 shape 一样，不论是图像还是 label，或者 box，都必须一致了。**

(2) 尺寸的原因。**检查卷积核的尺寸和输入尺寸是否匹配，padding数是否正确**。

**4. invalid argument 0: Sizes of tensors must match except in dimension 1. Got 14 and 13 in dimension 0 at /home/prototype/Downloads/pytorch/aten/src/THC/generic/THCTensorMath.cu:83**

(1)  **你输入的图像数据的维度不完全是一样的**，比如是训练的数据有 100 组，其中 99 组是 256*256，但有一组是 384*384，这样会导致Pytorch的检查程序报错。 -> 整理一下你的数据集保证每个图像的维度和通道数都一致即可。

(2)  另外一个则是比较隐晦的 batchsize 的问题，Pytorch 中检查你训练维度正确是按照每个 batchsize 的维度来检查的。 如果你有 999 组数据，你继续使用 batchsize 为 4 的话，这样 999 和 4 并不能整除，你在训练前 249 组时的张量维度都为(4,3,256,256) 但是最后一个批次的维度为 (3,3,256,256)，Pytorch 检查到 (4,3,256,256) != (3,3,256,256)，维度不匹配，自然就会报错了，这可以称为一个小 bug。 -> 挑选一个可以被数据集个数整除的batchsize或者直接把batchsize设置为1即可。

**5. expected CPU tensor (got CUDA tensor)**

期望得到CPU类型张量，得到的却是CUDA张量类型。 很典型的错误，例如计算图中有的参数为cuda型有的参数却是cpu型就会遇到这样的错误。



### 三. version

**1.** **IndexError:** **invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number**

~~~python
total_loss += loss.data[0]
~~~

**A:** 这个错误很常见, 由于高版本的pytorch对接口进行更改的导致的! 

It's likely that you're using a more recent version of pytorch than we specified. Replacing this line with

~~~python
total_loss += loss.item()
~~~

Should probably solve the problem.

**2.** /home/zhaozhichao/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: **FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.**

**A：**进入错误所在文件的位置: 将 `np.dtype([("quint8", np.uint8, 1)])` 修改为 `np.dtype([("quint8", np.uint8, (1,))])`　即可。

**3.** UserWarning: **indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool** 

**A:** 项目目录: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py#L191

在 model.py 中 191 行添加如下内容:

~~~python
# obj_mask 转为 bool
obj_mask = obj_mask.bool()  # convert int8 to bool
noobj_mask = noobj_mask.bool() # convert int8 to bool
~~~

**4.** **warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")**

A: 不要再使用  `torch.nn.Functional` 下面的激活函数了。 比如 `F.Sigmoid(x)` 这种形式了。 正确的使用方式是:

~~~
sigmoid = torch.nn.Sigmoid() out = sigmoid(x)
~~~

**5**  KeyError: 'unexpected key "module.bn1.num_batches_tracked" in state_dict'

经过研究发现，在 pytorch 0.4.1 及后面的版本里，BatchNorm 层新增了 num_batches_tracked 参数，用来统计训练时的 forward 过的 batch 数目，源码如下（pytorch0.4.1）：

~~~python
if self.training and self.track_running_stats:
    self.num_batches_tracked += 1
    if self.momentum is None:  # use cumulative moving average
        exponential_average_factor = 1.0 / self.num_batches_tracked.item()
    else:  # use exponential moving average
        exponential_average_factor = self.momentum
~~~

大概可以看出，这个参数和训练时的归一化的计算方式有关。

因此，我们可以知道该错误是由于训练和测试所用的 pytorch 版本( 0.4.1 版本前后的差异)不一致引起的。具体的解决方案是：如果是模型参数（ Orderdict 格式，很容易修改）里少了 num_batches_tracked 变量，就加上去，如果是多了就删掉。偷懒的做法是将 load_state_dict 的 strict 参数置为 False，如下所示：  

~~~python
load_state_dict(torch.load(weight_path), strict=False)
~~~



### 四. Data & dataloader

**1. 在开始训练的时候 jupyter notebook　kernel 死掉的现象**

A:　最首先的应该是去检查　DataLoader 中的num_workers, 尝试一下将 num_workers　设置更小或者设置　`num_workers = 0`。      //  把 jupyter 改为 code 形式会有报错信息

**2. ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)**

​    出现这个错误的情况是，在服务器上的 docker 中运行训练代码时，batch size 设置得过大，shared memory 不够（因为docker 限制了 shm ）.解决方法是，将 Dataloader 的 num_workers 设置为 0.

**3. Assertion `cur_target >= 0 && cur_target < n_classes’ failed.**

**A: RuntimeError: cuda runtime error (59) : device-side assert triggered at /home/loop/pytorch-master/torch/lib/THC/generic/THCTensorMath.cu:15**

我们在分类训练中经常遇到这个问题，一般来说在我们网络中输出的种类数和你label设置的种类数量不同的时候就会出现这个错误。

但是，Pytorch有个要求，**在使用** **CrossEntropyLoss** **这个函数进行验证时label必须是以0开始的**：

假如我这样:

~~~
self.classes = [0, 1, 2, 3]
~~~

我的种类有四类，分别是0.1.2.3，这样就没有什么问题，但是如果我写成：

~~~
self.classes = [1, 2, 3, 4]
~~~

这样就会报错

**->** 可以判断为 pytorch 所设计的分类器的分类 label 为 `[0,max-1]`，而 true ground 的标签值为 `[1,max]`。 所以可以通过修改 `label = (label-1).to(opt.device)`



### 五. 模型载入问题

**1. RuntimeError: Error(s) in loading state_dict for Missing key(s) in state_dict: “fc.weight”, “fc.bias”.**

像这种出现**丢失 key**(**missing key**)

If you have partial state_dict, which is missing some keys you can do the following:

~~~python
state = model.state_dict()
state.update(partial)
model.load_state_dict(state)
~~~

**2. RuntimeError: Error(s) in loading state_dict for Missing key(s) in Unexpected key(s) in state_dict: “classifier.0.weight”,**

**A:** 

~~~
# original saved file with DataParallel
state_dict = torch.load('myfile.pth.tar')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
~~~



### 六. 损失函数 

**1. 训练时损失出现 nan** 

可能导致梯度出现 nan 的三个原因：

1). **梯度爆炸**。也就是说梯度数值超出范围变成 nan. **通常可以调小学习率、加 BN 层或者做梯度裁剪来试试看有没有解决**。

2). **损失函数或者网络设计**。比方说，**出现了除 0，或者出现一些边界情况导致函数不可导，比方说 log(0)、sqrt(0)**.

3). 脏数据。**可以事先对输入数据进行判断看看是否存在nan.**

补充一下nan数据的判断方法：

注意！像nan或者inf这样的数值不能使用 == 或者 is 来判断！ 为了安全起见统一使用 `math.isnan()` 或者 `numpy.isnan()` 吧。

例如：

~~~
import numpy as np
 
# 判断输入数据是否存在nan
if np.any(np.isnan(input.cpu().numpy())):
  print('Input data has NaN!')
 
# 判断损失是否为nan
if np.isnan(loss.item()):
  print('Loss value is NaN!')
~~~

**2. pytorch中loss函数的参数设置**

​    以 CrossEntropyLoss 为例：

~~~
 CrossEntropyLoss(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean') 
~~~

- **reduce**：

​    若 **reduce = False**，那么 size_average 参数失效，直接**返回向量形式**的 loss，即batch中每个元素对应的loss.

​    若 **reduce = True**，那么 loss 返回的是标量：

​        如果 size_average = True，返回 **loss.mean()**.

​        如果 size_average = False，返回 **loss.sum()**.

- **weight** : 输入一个1D的权值向量，**为各个类别的loss加权**。

- **ignore_index** : 选择要忽视的目标值，使其对输入梯度不作贡献。如果 size_average = True，那么只计算不被忽视的目标的loss的均值。
- **reduction** : 可选的参数有：‘none’ | ‘elementwise_mean’ | ‘sum’, 正如参数的字面意思，不解释。

**七. 多机多卡与并行**

**1. 含有: torch.nn.DataParallel  的代码无法在 CPU上运行**

解决方案， 使用如下代码替换 DataParallel  进行一次封装:

~~~python
class WrappedModel(torch.nn.Module):
    def __init__(self, module):
            super(WrappedModel, self).__init__()
            self.module = module   # that I actually define.
    def forward(self, *x, **kwargs):
            return self.module(*x, **kwargs)
~~~

**Why：**

含有 DataParallel 的代码， 会对源代码添加一层  module 进行封装， 我们所做的仅仅是对其进行了一次 module 的封装

**2. 使用nn.Dataparallel 数据不在同一个gpu 上**

背景：pytorch 多GPU训练主要是采用数据并行方式：

~~~python
model = nn.DataParallel(model) 
~~~

问题：但是一次同事训练基于光流检测的实验时发现 data not in same cuda, 做代码 review 时候，打印每个节点 tensor，cuda 里的数据竟然没有分布在同一个 gpu 上

->  最终解决方案是在数据，吐出后统一进行执行.cuda() 将数据归入到同一个 cuda 流中解决了该问题。

**3. pytorch model load可能会踩到的坑：**

如果使用了nn.Dataparallel 进行多卡训练在读入模型时候要注意加.module， 代码如下:

~~~python
def get_model(self):
  if self.nGPU == 1:         
      return self.model     
  else:         
      return self.model.module
~~~

**4. 多 GPU 的处理机制**

使用多 GPU 时，应该记住 pytorch 的处理逻辑是：

1) 在各个 GPU 上初始化模型。

2) 前向传播时，把 batch 分配到各个 GPU 上进行计算。

3) 得到的输出在主 GPU 上进行汇总，计算 loss 并反向传播，更新主 GPU上的权值。

4) 把主 GPU 上的模型复制到其它 GPU 上。



### 八. 优化问题(lr & optim)

**1. 优化器的 weight_decay 项导致的隐蔽 bug**

​        我们都知道 weight_decay 指的是权值衰减，即在原损失的基础上加上一个 L2 惩罚项，使得模型趋向于选择更小的权重参数，起到正则化的效果。但是我经常会忽略掉这一项的存在，从而引发了意想不到的问题。 这次的坑是这样的，在训练一个 ResNet50 的时候，网络的高层部分 layer 4暂时没有用到，因此也并不会有梯度回传，于是我就放心地将 ResNet50 的所有参数都传递给 Optimizer 进行更新了，想着 layer4 应该能保持原来的权重不变才对。但是实际上，尽管 layer4 没有梯度回传，但是weight_decay 的作用仍然存在，它使得 layer4 权值越来越小，趋向于 0。后面需要用到 layer4 的时候，发现输出异常（接近于0），才注意到这个问题的存在。 虽然这样的情况可能不容易遇到，但是还是要谨慎：**暂时不需要更新的权值，一定不要传递给 Optimizer，避免不必要的麻烦**。



### 九. pytorch 可重复性问题

https://blog.csdn.net/hyk_1996/article/details/84307108



### 十. 基本语法问题

**1. RuntimeError: some of the strides of a given numpy array are negative. This is currently not supported, but will be added in future releases.**

~~~python
import numpy as np
x = np.random.random(size=(32, 32, 7))
torch.from_numpy(np.flip(x, axis=0))
~~~

Same error with np.rot90()

**A:** ndarray.copy() will alocate new memory for numpy array which make it normal, I mean the stride is not negative any more.

~~~python
torch.from_numpy(np.flip(x, axis=0).copy())
~~~

**2. view()操作只能用在连续的tensor下**

利用 `is_contiguous()` 判断该 tensor 在内存中是否连续，不连续的话使用 `.contiguous()`使其连续。

**3. input is not contiguous at /pytorch/torch/lib/THC/generic/THCTensor.c:227**

~~~
batch_size, c, h, w = input.size()
 rh, rw = (2, 2)
 oh, ow = h * rh, w * rw
 oc = c // (rh * rw)
 out = input.view(batch_size, rh, rw, oc, h, w)
 out = out.permute(0, 3, 4, 1, 5, 2)
 out = out.view(batch_size, oc, oh, ow)
invalid argument 2: input is not contiguous at /pytorch/torch/lib/THC/generic/THCTensor.c:227
~~~

**A:** 上述在第7行报错，报错原因是由于浅拷贝。上面式子中input为Variable变量。

上面第5行 `out = out.permute(0, 3, 4, 1, 5, 2)`  时执行了浅拷贝，out 只是复制了out 从 input 传递过来的指针，也就是说 input 要改变 out 也要随之改变。

解决方法是，在第6行的时候使用 `tensor.contiguous()`，第6行改成:`out = out.permute(0, 3, 4, 1, 5, 2).contiguous()`即可。

**4. RuntimeError: some of the strides of a given numpy array are negative. This is currently not supported, but will be added in future releases.**

**A:** 这个原因是因为程序中操作的 numpy 中有使用负索引的情况：`image[…, ::-1]`。

解决办法比较简单，加入 image 这个 numpy 变量引发了错误，返回 `image.copy()` 即可。因为copy操作可以在原先的numpy变量中创造一个新的不适用负索引的numpy变量。

**5. torch.Tensor.detach()的使用**

`detach()`的官方说明如下： Returns a new Tensor, detached from the current graph. The result will never require gradient. 

**假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B. 那么可以这样做：** 

~~~python
input_B = output_A.detach()
~~~

它可以使两个计算图的梯度传递断开，从而实现我们所需的功能。

**6. ValueError: Expected more than 1 value per channel when training**

当 batch 里只有一个样本时，再调用 batch_norm 就会报下面这个错误： **raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))**。 没有什么特别好的解决办法，在训练前用 **num_of_samples % batch_size** 算一下会不会正好剩下一个样本。 **! 当 bacthsize 为1的时候, batchnorm 是无法运行的。**



**十一. 基本用法**

**Q: Pytorch 如何忽略警告**

~~~python
python3 -W ignore::UserWarning xxxx.py
~~~

