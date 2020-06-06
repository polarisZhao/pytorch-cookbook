# PyTorch_ONNX_TensorRT
**!!! 生成的模型， 必须由相同的算力的显卡来解析, 具体的算力可以在 https://developer.nvidia.com/cuda-gpus 查看。**

## 安装 TensorRT

#####  这里建议采用　tar　包进行安装，否则会带了一些未知错误 !

1. 首先你需要安装好 **NVIDIA 驱动**、**cuda**、**cudnn**。

2. 在 https://developer.nvidia.com/nvidia-tensorrt-7x-download下载  **tensorRT ７.0** 的安装包，选择对应的 cuda 和 cudnn 版本，　这里我们选择 **Tar File Install Packages For Linux x86**　下面的　**TensorRT 7.0.0.11 for Ubuntu 18.04 and CUDA 10.0 tar package**  并解压。

3. 在 `vim ~/.bashrc`，添加如下内容

   ~~~shell
   export LD_LIBRARY_PATH=[/path/to/TensorRT]/lib:$LD_LIBRARY_PATH　#　[/path/to/TensorR] 替换成你自己实际的目录
   ~~~

​       执行　`source ~/.bashrc` 使其生效。

4. 执行如下命令，安装对应的 **Python** 模块。

   ~~~shell
   $ cd python/ # TensoRT
   $ pip3 install tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl 
   $ cd ../uff/ # uff
   $ pip3 install uff-0.6.5-py2.py3-none-any.whl 
   $ cd ../graphsurgeon/   # graphsurgeon
   $ pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl 
   ~~~

5. 编译 sample。

   ~~~
   $ cd sample
   $ make
   ~~~



## TensorRT workFlow

### 1. Workflow TensorRT

**Input:**  Pre-tained FP32 model and network

**Output:**  Optimized execution engine on GPU for deployment

​    **Serialized a plan can be reloaded from the disk into TensorRT tuntime. There is no need to perform the optimization step again**

### 2. Three Step

##### (1)  model Parser

可以有两种方式对模型进行解析：

- **Parser** 方式，即模型解析器，解析出其中的网络层及网络层之间的连接关系，然后将其输入到 TensorRT 中。

  Parser 目前有三个：

  - Caffe Parser
  - Uff，这个是 NV 定义的网络模型的一种文件结构，现在 TensorFlow 可以直接转成 uff
  - onnx parser，是Facebook主导的开源的可交换的各个框架都可以输出的。

- API 接口。允许用户自定义层。 目前API支持两种接口实现方式，一种是 C++，另一种是 Python，Python 接口可能在一些快速实现上比较方便一些。

~~~python
with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_network(EXPLICIT_BATCH) as network, \
     trt.OnnxParser(network, TRT_LOGGER) as parser: # ！声明时 parser 就和 network 绑定
      
    if not os.path.exists(onnx_file_path):
        quit('ONNX file {} not found'.format(onnx_file_path))

    print('Loading ONNX file from path {}...'.format(onnx_file_path))
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read()) # ！这里将其存储到 network 中
    print('Completed parsing of ONNX file')
~~~

##### (2) Engine optim

第一，网络层进行的合并。大家如果了解GPU的话会知道，在GPU上跑的函数叫Kernel，TensorRT是存在Kernel的调用的。

- 在绝大部分框架中，比如一个卷积层、一个偏置层和一个relu层，这三层是需要调用三次cuDNN对应的API，但实际上conv、bias和relu这三层的实现完全是可以合并到一起的，TensorRT会对一些可以合并网络进行合并
- 目前的网络一方面越来越深，另一方面越来越宽，可能并行做若干个相同大小的卷积，这些卷积计算其实也是可以合并到一起来做的

第二，比如在 concat 这一层，比如说这边计算出来一个 1×3×24×24，另一边计算出来 1×5×24×24，conca t到一起，变成一个 1×8×24×24 的矩阵，这个叫 concat 这层这其实是完全没有必要的，因为TensorRT完全可以实现直接接到需要的地方，不用专门做 concat 的操作，所以这一层也可以取消掉。

第三，Kernel可以根据不同的 batch size 大小和问题的复杂程度，去选择最合适的算法，TensorRT预先写了很多GPU实现，有一个自动选择的过程。

第四，不同的 batch size 会做 tuning。

第五，不同的硬件如P4卡还是V100卡甚至是嵌入式设备的卡，TensorRT都会做优化，得到优化后的engine。

~~~python
engine = builder.build_cuda_engine(network)
~~~

另外，**Engine可以序列化到内存（buffer）或文件（file）**

~~~python
with open(engine_file_path, "wb") as f:
    f.write(engine.serialize())
~~~

##### (3) Execution

读取模型并反序列化，将其变成 engine 以供使用。

~~~python
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
~~~

然后在执行的时候创建 context，主要是分配预先的资源

~~~python
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """ 为 engine 中的每个 binding 都申请 host(CPU) 和 device(GPU) buffer
        并绑定输入和输出
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

context = engine.create_execution_context()
~~~

Engine 加 context 就可以做推断（Inference）。

~~~python
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
  
inputs[0].host = ...
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
~~~

## Bugs

**1. AttributeError: ‘NoneType’ object has no attribute ‘create_execution_context’**

~~~python
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
~~~

**2. pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?**

原因： pycuda.driver 没有初始化，导致无法得到 context，需要在导入 pycuda.driver 后再导入 pycuda.autoinit,  即如下：

~~~
import pycuda.driver as cuda
import pycuda.autoinit
~~~

**3. output tensor has no attribute _trt**

模型中有一些操作还没有实现，需要自己实现。



## Credits

- https://github.com/zerollzeng/tiny-tensorrt
- https://github.com/NVIDIA-AI-IOT/torch2trt
- https://github.com/Rapternmn/PyTorch-Onnx-Tensorrt
- https://github.com/onnx/onnx-tensorrt



## TBD

- [ ] TensorRT 对齐操作(int8量化)  

- [ ] 如何添加一个新层

- [ ] 跑通YOLOv3的检测流程

- [ ] C++ 调用

  