from tqdm import tqdm
import pandas as pd
import time
import csv
import os
import numpy as np


# 功能：ix2tag
def get_ix_to_tag(dictionary):
    ix_to_tag = {}
    for key in dictionary:
        ix_to_tag[dictionary[key]] = key
    return ix_to_tag


# 全局变量
START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING = "<PAD>"
SBME_tag_to_ix = {"S": 0, "B": 1, "M": 2, "E": 3, PADDING: 4, START_TAG: 5, STOP_TAG: 6}
SBME_ix_to_tag = get_ix_to_tag(SBME_tag_to_ix)
BI_tag_to_ix = {"B": 0, "I": 1, PADDING: 2, START_TAG: 3, STOP_TAG: 4}
BI_ix_to_tag = get_ix_to_tag(BI_tag_to_ix)


# 功能：读取已经分好词的训练集 并且给其打上SBME 4tag标注/ BI 2tag标注
# 进一步：对句子做padding 使得其能够使用mini-batch
def readCorpus(_path, mode="SBME"):
    f = open(_path, 'r', encoding='utf-8')
    x, y = [], []
    for line in f:
        x_line = line.split()  # 分好的各个词
        x_meta = []  # 存储单个letter
        y_meta = []  # 存储对应letter的tag
        # 打上tag
        for word in x_line:
            if len(word) == 1:
                x_meta.append(word)
                y_meta.append('S')
            else:
                first = True
                for i, letter in enumerate(word):
                    if first:
                        y_meta.append('B')
                        first = False
                    elif i == len(word) - 1:
                        y_meta.append('E')
                    else:
                        y_meta.append('M')
                    x_meta.append(letter)
        # endfor
        assert len(x_meta) == len(y_meta)
        # print(x_meta)
        # print(y_meta)
        if len(x_meta) > 0:
            x.append(x_meta)
            y.append(y_meta)
    # endfor
    return x, y


# 功能：将SBME标记法转换为BI标记
def SBME2BI(tags):
    BItags = []
    for tag in tags:
        if tag == "S" or tag == "B":
            BItags.append('B')
        else:
            BItags.append("I")
    return BItags


STOP_WORD = '<unk>'


# 功能：根据tr_x获取词的编码字典
def getWord2Ix(tr_x):
    # 将每个词按照出现的顺序映射到相应数字
    # 注：tags未用
    word2Ix = {}
    for sentence in tqdm(tr_x):
        for word in sentence:
            if word not in word2Ix:
                word2Ix[word] = len(word2Ix)
    word2Ix[STOP_WORD] = len(word2Ix)  # 代表停用词
    return word2Ix


# 封装停用词处理的字典
class word2Idx:
    def __init__(self, tr_x):
        self.dict = getWord2Ix(tr_x)

    def __getitem__(self, key):
        if key in self.dict:
            return self.dict[key]
        else:  #
            return self.dict[STOP_WORD]


# 功能：获取预训练的词向量字典
def getWord2Vec(vecPath="../data/word2vec/sgns.renmin.bigram-char"):
    begin_time = time.time()
    word2Vec = pd.read_table(vecPath, sep=" ", index_col=0, quoting=csv.QUOTE_NONE,
                             encoding="utf-8", header=None, skiprows=1, usecols=range(301))
    end_time = time.time()
    print("Load embedding consumed: {:.2f}".format(end_time - begin_time))
    return word2Vec


# 功能：通过数据集的word_to_idx得到本数据集对应的词向量矩阵
def getEmbedding(word_to_idx, embedding_mat_path, emb_dim=300):
    # 未存储对本训练集的embedding
    if not os.path.isfile(embedding_mat_path):
        matrix_len = len(word_to_idx)
        weights_matrix = np.zeros((matrix_len, emb_dim))
        words_found = 0
        # 获取词向量字典
        word2vec = getWord2Vec()
        for word in word_to_idx:
            idx = word_to_idx[word]
            try:
                weights_matrix[idx] = word2vec.loc[word]
                words_found += 1
            except KeyError:  # 词向量字典中不存在此单词——比如手动添加的停用词
                # 采用随机初始化
                weights_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))
        np.save(embedding_mat_path, weights_matrix)
        return weights_matrix
    else:
        return np.load(embedding_mat_path)


if __name__ == "__main__":
    # x, y = readCorpus("../data/corpus/msr_test_gold.utf8")
    # word2iX = word2Idx(x)
    # v = getWord2Vec()
    x, y = readCorpus("../data/corpus/msr_training.utf8")
    s = [len(xx) for xx in x]
