# Task 1 入门任务

## NNI 体验文档

### 1. AutoML 工具比较
机器学习算法与模型的选择，对机器学习十分重要，一个成功的选择，能够成倍提高训练效率，从而提高模型准确度，减少损失，产生更大的效益。

但算法与模型的选择并不简单。就算是数据科学家，也需要花费大量的时间用于尝试与权衡不同模型的优劣，最终才能得出理想的结果。超参的调参过程中也经常造成算力的浪费。

自动机器学习（AutoML）是一套自动化的机器学习应用工具，旨在用自动化工具完成特征工程、自动调参等优化工作。

当前，自动机器学习平台早已问世，下面介绍几个著名的AutoML工具，并列出优缺点，以供比较。

* Microsoft NNI

NNI（Neural Network Intelligence）是微软开源的自动机器学习工具，面向研究人员和算法工程师而设计，2018年9月问世，目前已经更新至v2.0。

支持多平台，支持命令行操作，支持结果可视化。内置优化算法多，功能丰富，扩展性强，支持远程调用进行集群训练。

* PyBrain

PyBrain是一个Python的模块化机器学习库，其中包含用于神经网络的算法，如强化学习、无监督学习和进化等，能够为机器学习任务和各种预定义环境提供灵活、强大的算法支撑。

该库适用于处理涉及连续状态和动作空间的多维问题，其使用函数逼近器即神经网络将所有训练方法作为待训练实例。且PyBrain非常易用，不仅适合入门人员也能够为研究提供灵活、高效的算法。

* Milk

Milk同样是Python的机器学习工具包，其偏向于有多种分类器的监督分类，如SVM(基于libsvm)、k-NN、随机森林、决策树、自我组织地图、特征选择的分步判别分析、亲和力传播等，这些分类器还可以以多种方式组成不同的分类系统。

其输入相当灵活，支持数值库numpy以及很多的基本库，非常强调速度和低内存使用，大部分代码都是以C++方式实现的。

* Amazon SageMaker

Amazon SageMaker是一个完全托管的开源学习工具，能够提供快速构建、训练和部署机器学习的能力，同时开发者能够在一个集成的可视化界面中编写、跟踪代码、可视化数据以及进行调试和监控等。

其具有完整的平台IDE、具体代码与API，简洁易用，这是第一个用于机器学习的完全集成式开发环境 (IDE)  单一集成的可视界面操作。

### 2. NNI 安装及使用

NNI的安装非常简单，只需一行命令即可：

```
$ pip install --upgrade nni
```
![](./code/安装.jpg)

可视化的web UI

![](./code/使用.jpg)

### 3. NNI 使用感受

安装和使用都比较方便，作为入门来说很有学习价值。

## NNI 样例分析文档

### 配置文件：experiment_config.yml

```
authorName: default
experimentName: example_mnist_pytorch
trialConcurrency: 1
maxExecDuration: 2h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python mnist.py
  codeDir: .
  gpuNum: 0
```

### 搜索空间：search_space.json

```json
{
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    "momentum":{"_type":"uniform","_value":[0, 1]}
}
```
### 代码

代码部分只需要在原有PyTorch代码上进行些许修改。

1. 参数选择无需在程序中给定，而是通过nni获得：
```python
tuner_params = nni.get_next_parameter()
```
获得的参数是一个dict对象，通过搜索空间定义的名称可索引出对应的参数值。

1. 训练中途，报告中间结果：
```python
nni.report_intermediate_result(test_acc)
```
可在间隔若干个epoch后报告中间结果，也可在间隔若干时间后报告中间结果。

3. 在训练完整结束后，报告最终结果：
```python
nni.report_final_result(test_acc)
```
报告的最终结果作为训练的default metric，用于不同trial之间的比较。

#### 结果

如图，10次trial都成功地完成，其中id为9的trial达到了最高准确率，达99.34%。

![](./Images/1.png)

![](./images/4.png)

#### 超参组合可视化

![](./images/5.png)

图中，准确率更高的组合用红线表示，而准确率低的用绿线表示。

可以看出，当batch_size选择16，lr和momentum大小适中时，模型可以达到99%以上的准确率，实验效果非常理想。

#### 训练结果可视化

![](./images/3.png)

![](./images/2.png)

![](./images/6.png)

![](./images/7.png)
