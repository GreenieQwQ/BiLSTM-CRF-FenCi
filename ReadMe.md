# 文件说明

## 预处理相关代码

### 使用预训练词向量

使用预训练词向量相关函数可见于'preprocess.py'文件。

### 分割数据集、对数据集作padding

通过重载dataloader与dataset类来完成目的，相关代码可见于'data.py'文件。

### 读入数据集、进行标注

读取数据集并进行SBME标注相关函数可见于'preprocess.py'文件。

## 模型相关代码

### Blstm-Crf模型

Blstm-Crf模型及训练相关代码可见于'model.py'文件中。

### Blstm、Lstm模型

Blstm、Lstm模型及训练相关代码可见'blstm.py'及'lstm.py'代码文件。

## 辅助函数

评估、根据tag生成分词等辅助函数可见'utils.py'文件。

# 运行说明

本实验运行时文件的组织结构如下：

```
├─data
│  │  corpus 
|  |    |  msr_test_gold.utf8
|  |    |  msr_training.utf8
│  │  model
|  |  word2vec
|  |    |  sgns.renmin.bigram-char
│
├─src
│  │  blstm.py
│  │  data.py
│  │  utils.py
│  │  lstm.py
|  |  model.py
│  |  preprocess.py
|  
```

sgns.renmin.bigram-char预训练词向量文件可在https://github.com/Embedding/Chinese-Word-Vectors下载。

# 运行环境

- python 3.8
- pytorch==1.7.0
- cudatoolkit==10.2

# 依赖安装

使用语句

```
pip install -r requirement.txt
```

即可安装除pytorch外的依赖项。pytorch安装方法可见于https://pytorch.org/get-started/locally/官方网址。

# 运行说明

安装上述依赖、确保文件组织正确，即可使用命令

```
python model.py 200 --usecuda
```

训练batch\_size=200的模型。更多运行参数可使用命令

```
python model.py -h
```

查看。使用命令

```
python lstm.py
python blstm.py
```

即可运行blstm、lstm模型。

# 其余模型参数

其余报告中训练的模型参数可在https://github.com/GreenieQwQ/nlp_fenci_parameters下载。