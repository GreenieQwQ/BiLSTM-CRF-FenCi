
## 预处理相关代码

### 将数据集提供的xml文件转化为IOB数据

’xml2IOB.py‘代码包含了将数据集提供的xml文件转化为IOB数据的相关函数。

### 获取Glove词向量

加载Glove词向量相关代码为'word2vec.py'文件。

## 模型相关代码

### Blstm-Crf模型

Blstm-Crf模型及训练相关代码可见于'model.py'文件中。

### Blstm、Lstm模型

Blstm、Lstm模型及训练相关代码可见'blstm.py'及'lstm.py'代码文件。

# 运行说明

需要将文件按照如下方式组织、并且存放相关数据后即可运行相关代码文件：

- src——存放代码的文件夹

- data——存放相关数据的文件夹

  - model——存放模型的文件夹
    - lstm——存放lstm模型数据的文件夹
    - blstm——存放blstm模型数据的文件夹
    - blstm-crf——存放blstm-crf模型数据的文件夹

  - SemEval——存放从官网下载的训练集、测试集的文件夹

  - glove——存放从glove官网加载的txt词向量文件的文件夹

# 运行环境

- python 3.8
- pytorch==1.7.0

# Author

By 18340206