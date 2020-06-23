# PyTorch_ONNX_TensorRT
- [ ] 跑通一个 分类的程序 + cpp 调用代码 ! !

- [ ] 搞明白 tensort 的基本逻辑

- [ ] tensorRT 对齐操作(int8量化)  

- [ ] 如何添加一个新层

- [ ] 跑通一个检测流程

- [ ] 整理 README 文档(安装、两个示例程序、辅助脚本、常见错误)

  

**!!! 生成的模型， 必须由相同的算力的显卡来解析, 具体的算力可以在 https://developer.nvidia.com/cuda-gpus 查看。**

## 安装 TensorRT

>Python 环境:
>
>*.pth[pytorch] > *.onnx[onnx] > *.trt[tensorRT] 
>
>*.pth[pytorch] > *.txtp[TensorRT] Python 环境



## TensorRT 执行流程

- Create Builder : 包含 TensorRT 组件、pineline、buffer地址、输入输出维度

- Create Network : 保存训练好的神经网络、其输入是神经网络模型(onnx、tf)， 其输出是可执行的推理引擎。

- Create Parser: 解析网络

- Binging input、output　以及自定义组件

- 序列化或者反序列化

- 传输计算数据(host->device)

- 执行计算

- 传输计算结果(device->host)



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



## BAK

报错：UnicodeDecodeError: ‘utf-8’ codec can’t decode byte 0xaa in position 8: invalid start byte
原因：在打开导入序列化模型时，需要采用’rb’模式才能读，否则不能读取，即在读取序列化模型时，需要做3件事，如下：

打开文件，必须用rb模式：with open(cfg.work_dir + ‘serialized.engine’, ‘rb’) as f
创建runtime：trt.Runtime(logger) as runtime
基于runtime生成反序列化模型：engine = runtime.deserialize_cuda_engine(f.read())
报错：onnx.onnx_cpp2py_export.checker.ValidationError: Op registered for Upsample is deprecated in domain_version of 11
附加报错信息：
Context: Bad node spec: input: “085_convolutional_lrelu” output: “086_upsample” name: “086_upsample” op_type: “Upsample” attribute
{ name: “mode” s: “nearest” type: STRING } attribute { name: “scales” floats: 1 floats: 1 floats: 2 floats: 2 type: FLOATS }
问题原因：onnx更新太快了，在官方1.5.1以后就取消了upsample层，所以对yolov3报错了。参考https://devtalk.nvidia.com/default/topic/1052153/jetson-nano/tensorrt-backend-for-onnx-on-jetson-nano/1
修改方式是降级onnx到1.4.1
pip uninstall onnx
pip install onnx==1.4.1





