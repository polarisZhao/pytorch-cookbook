# pytorch-cookbook

本文代码基于PyTorch 1.0版本，需要用到以下包

~~~
import collections
import os
import shutil
import tqdm

import numpy as np
import PIL.Image
import torch
import torchvision
~~~

#### 1. 基础配置

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

#### 2. 张量处理

##### (1) 张量的基本信息

~~~
tensor.type()   # Data type
tensor.size()   
# Shape of the tensor. It is a subclass of Python    tuple
tensor.dim()    # Number of dimensions.
~~~

##### (2) 数据类型转换

~~~
# Set default tensor type. Float in PyTorch is much faster than double.
torch.set_default_tensor_type(torch.FloatTensor)

# Type convertions.
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()

torch.Tensor与np.ndarray转换
# torch.Tensor -> np.ndarray.
ndarray = tensor.cpu().numpy()

# np.ndarray -> torch.Tensor.
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float()  # If ndarray has negative stride
~~~

##### (3)  torch.Tensor 与 PIL.Image 转换

PyTorch中的张量默认采用N×D×H×W的顺序，并且数据范围在[0, 1]，需要进行转置和规范化。

~~~
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor * 255, min=0, max=255
    ).byte().permute(1, 2, 0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way

# PIL.Image -> torch.Tensor.
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))
    ).permute(2, 0, 1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))  # Equivalently way

np.ndarray与PIL.Image转换
# np.ndarray -> PIL.Image.
image = PIL.Image.fromarray(ndarray.astypde(np.uint8))

# PIL.Image -> np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))
~~~

##### (4) 从只包含一个元素的张量中提取值

这在训练时统计loss的变化过程中特别有用。否则这将累积计算图，使GPU存储占用量越来越大。

~~~
value = tensor.item()
~~~


##### (5) 张量形变

张量形变: 张量形变常常需要用于将卷积层特征输入全连接层的情形。相比torch.view，torch.reshape可以自动处理输入张量不连续的情况。


~~~
tensor = torch.reshape(tensor, shape)
~~~

##### (6) 打乱顺序

~~~
    # Shuffle the first dimension
    tensor = tensor[torch.randperm(tensor.size(0))]  
~~~

##### (7) 复制张量: 有三种复制的方式，对应不同的需求。

| Operation             | New/Shared memory | Still in computation graph |
| --------------------- | ----------------- | -------------------------- |
| tensor.clone()        | New               | Yes                        |
| tensor.detach()       | Shared            | No                         |
| tensor.detach.clone() | New               | No                         |

##### (8) 拼接张量

注意`torch.cat`和`torch.stack`的区别在于`torch.cat`沿着给定的维度拼接，而`torch.stack`会新增一维。例如当参数是3个10×5的张量，`torch.cat`的结果是30×5的张量，而`torch.stack`的结果是3×10×5的张量。

~~~
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)
~~~

##### (9) 将整数标记转换成独热（one-hot）编码  
 (PyTorch中的标记默认从0开始)

~~~
   N = tensor.size(0)
   one_hot = torch.zeros(N, num_classes).long()
   one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
~~~

##### (10)得到非零/零元素

~~~
   torch.nonzero(tensor)               # Index of non-zero elements
   torch.nonzero(tensor == 0)          # Index of zero elements
   torch.nonzero(tensor).size(0)       # Number of non-zero elements
   torch.nonzero(tensor == 0).size(0)  # Number of zero elements
~~~

##### (11)张量扩展

~~~
   # Expand tensor of shape 64*512 to shape 64*512*7*7.
   torch.reshape(tensor, (64, 512, 1, 1)).expand(64, 512, 7, 7)
~~~

##### (12)矩阵乘法

~~~
# Matrix multiplication: (m*n) * (n*p) -> (m*p).
result = torch.mm(tensor1, tensor2)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p).
result = torch.bmm(tensor1, tensor2)

# Element-wise multiplication.
result = tensor1 * tensor2
~~~

##### (13) 计算两组数据之间的两两欧式距离

~~~
# X1 is of shape m*d.
X1 = torch.unsqueeze(X1, dim=1).expand(m, n, d)
# X2 is of shape n*d.
X2 = torch.unsqueeze(X2, dim=0).expand(m, n, d)
# dist is of shape m*n, where dist[i][j] = sqrt(|X1[i, :] - X[j, :]|^2)
dist = torch.sqrt(torch.sum((X1 - X2) ** 2, dim=2))
~~~

#### 3. 模型定义

##### (1) 卷积层 

最常用的卷积层配置是：

```python
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
```

如果卷积层配置比较复杂，不方便计算输出大小时，可以利用如下可视化工具辅助: <https://ezyang.github.io/convolution-visualizer/index.html>

##### (2) GAP（Global average pooling）层
~~~
   gap = torch.nn.AdaptiveAvgPool2d(output_size=1)
~~~

##### (3) 多卡同步BN（Batch normalization）

当使用torch.nn.DataParallel将代码运行在多张GPU卡上时，PyTorch的BN层默认操作是各卡上数据独立地计算均值和标准差，同步BN使用所有卡上的数据一起计算BN层的均值和标准差，缓解了当批量大小（batch size）比较小时对均值和标准差估计不准的情况，是在目标检测等任务中一个有效的提升性能的技巧。

参见： [Synchronized-BatchNorm-PyTorch​github](vacancy/Synchronized-BatchNorm-PyTorch​github.com)

##### (4) 计算模型参数量[D]

~~~
# Total parameters                    
num_params = sum(p.numel() for p in model.parameters()) 
# Trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
~~~

类似Keras的model.summary()输出模型信息，参见[pytorch-summary​github](sksq96/pytorch-summary​github.com)

##### (5) 模型权值初始化[D]

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

##### (6) 部分层使用预训练模型

注意如果保存的模型是`torch.nn.DataParallel`，则当前的模型也需要是`torch.nn.DataParallel`。`torch.nn.DataParallel(model).module == model`。

~~~
   model.load_state_dict(torch.load('model,pth'), strict=False)
~~~

将在GPU保存的模型加载到CPU:

~~~
   model.load_state_dict(torch.load('model,pth', map_location='cpu'))
~~~

#### 4. 特征提取与微调

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

##### (3)其他预训练模型
[pretrained-models](Cadene/pretrained-models.pytorchgithub.com)

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

#### 5. 模型训练

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

##### (2) 训练基本代码框架

~~~
for t in epoch(80):
    for images, labels in tqdm.tqdm(train_loader, desc='Epoch %3d' % (t + 1)):
        images, labels = images.cuda(), labels.cuda()
        scores = model(images)
        loss = loss_function(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
~~~

##### (3)  label smothing

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

##### (4) Mixup

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

##### (5) 双线性汇合（bilinear pooling）

~~~
X = torch.reshape(N, D, H * W)                        # Assume X has shape N*D*H*W
X = torch.bmm(X, torch.transpose(X, 1, 2)) / (H * W)  # Bilinear pooling
assert X.size() == (N, D, D)
X = torch.reshape(X, (N, D * D))
X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)   # Signed-sqrt normalization
X = torch.nn.functional.normalize(X)                  # L2 normalization
~~~

##### (6) L1 正则化

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

##### (7) 不对偏置项进行L2正则化/权值衰减（weight decay）

~~~
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},                
              {'parameters': others_list}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
~~~

##### (8) 梯度裁剪（gradient clipping）

 ~~~
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
 ~~~

##### (9) 计算Softmax 输出的正确率

~~~
score = model(images)
prediction = torch.argmax(score, dim=1)
num_correct = torch.sum(prediction == labels).item()
accuruacy = num_correct / labels.size(0)
~~~

##### (10) 可视化模型前馈计算图：

https://github.com/szagoruyko/pytorchviz

##### （11）可视化学习曲线

有Facebook自己开发的Visdom和Tensorboard两个选择。
facebookresearch/visdomgithub.com
lanpa/tensorboardXgithub.com

~~~
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

##### （12）得到当前学习率

~~~
# If there is one global learning rate (which is the common case).
lr = next(iter(optimizer.param_groups))['lr']

# If there are multiple learning rates for different layers.
all_lr = []
for param_group in optimizer.param_groups:
    all_lr.append(param_group['lr'])
~~~

##### （13）学习率衰减

~~~python
# Reduce learning rate when validation accuarcy plateau.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
for t in range(0, 80):
    train(...); val(...)
    scheduler.step(val_acc)

# Cosine annealing learning rate.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
# Reduce learning rate by 10 at given epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
for t in range(0, 80):
    scheduler.step()    
    train(...); val(...)

# Learning rate warmup by 10 epochs.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: t / 10)
for t in range(0, 10):
    scheduler.step()
    train(...); val(...)
~~~

##### （14）保存与加载断点

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


#### 6. Pytorch 其他注意事项

##### (1) 模型定义

- 建议有参数的层和汇合（pooling）层使用`torch.nn`模块定义，激活函数直接使用 `torch.nn.functional`。`torch.nn`模块和`torch.nn.functional`的区别在于，`torch.nn`模块在计算时底层调用了`torch.nn.functional`，但`torch.nn`模块包括该层参数，还可以应对训练和测试两种网络状态。使用`torch.nn.functional`时要注意网络状态，如

~~~
def forward(self, x):
    ...
    x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
~~~

- `model(x)`前用 `model.train()`和 `model.eval()`切换网络状态。不需要计算梯度的代码块用 `with torch.no_grad()`包含起来。`model.eval()`和`torch.no_grad()`的区别在于，`model.eval()`是将网络切换为测试状态，例如BN和随机失活（dropout）在训练和测试阶段使用不同的计算方法。`torch.no_grad()`是关闭PyTorch张量的自动求导机制，以减少存储使用和加速计算，得到的结果无法进行`loss.backward()`。`torch.nn.CrossEntropyLoss`的输入不需要经过`Softmax`。

- `torch.nn.CrossEntropyLoss`等价于`torch.nn.functional.log_softmax` + `torch.nn.NLLLoss`。

- `loss.backward()`前用`optimizer.zero_grad()`清除累积梯度。

- `optimizer.zero_grad()`和`model.zero_grad()`效果一样。

##### (2) PyTorch性能与调试

- `torch.utils.data.DataLoader`中尽量设置`pin_memory=True`，对特别小的数据集如MNIST设置`pin_memory=False` 反而更快一些。
- `num_workers` 的设置需要在实验中找到最快的取值。
- 用`del`及时删除不用的中间变量，节约GPU存储。
- 使用`inplace`操作可节约 GPU 存储，如

 ~~~
x = torch.nn.functional.relu(x, inplace=True)
 ~~~

- 减少CPU和GPU之间的数据传输。例如， 如果你想知道一个 epoch 中每个 mini-batch 的 loss 和准确率，先将它们累积在 GPU 中等一个 epoch 结束之后一起传输回 CPU 会比每个 mini-batch 都进行一次 GPU 到 CPU 的传输更快。
- 使用半精度浮点数`half()`会有一定的速度提升，具体效率依赖于GPU型号。需要小心数值精度过低带来的稳定性问题。时常使用 `assert tensor.size() == (N, D, H, W)`作为调试手段，确保张量维度和你设想中一致。
- 除了标记 y 外，尽量少使用一维张量，使用n*1的二维张量代替，可以避免一些意想不到的一维张量计算结果。
- 统计代码各部分耗时

~~~python
with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as profile:
    ...
    print(profile)
~~~

或者在命令行运行：

~~~
python -m torch.utils.bottleneck main.py
~~~

参考链接：

- Tensorflow cookbook

- https://github.com/kevinzakka/pytorch-goodies

- https://github.com/chenyuntc/pytorch-book

- pytorch 官方文档和tutorial

