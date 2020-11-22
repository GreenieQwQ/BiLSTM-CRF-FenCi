# 参考代码的Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import enlighten
import numpy as np
import os
from xml2IOB import *
from word2vec import *
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, precision_score, accuracy_score, recall_score, classification_report
from preprocess import *
from utils import *

torch.manual_seed(1)
# 全局变量
START_TAG = "<START>"
STOP_TAG = "<STOP>"
SBME_tag_to_ix = {"S": 0, "B": 1, "M": 2, "E": 3, START_TAG: 4, STOP_TAG: 5}
SBME_ix_to_tag = get_ix_to_tag(SBME_tag_to_ix)
BI_tag_to_ix = {"B": 0, "I": 1, START_TAG: 2, STOP_TAG: 3}
BI_ix_to_tag = get_ix_to_tag(BI_tag_to_ix)


# 功能：选取行向量（非列向量）的最大下标
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


"""
    输入：seq：sentence
        to_ix：sentence对应词的字典 将每个词映射到一个index
    输出：将seq转换为index的tensor
    注：torch.long为整数
"""


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 功能：为前向计算而计算log sum exp的步骤
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    # 将max_score广播至所有维度
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # 返回结果 实质相当于log_sum_exp(vec)，与max_score无关
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    """
        输入：tag_to_ix：字典：将所有的标签转换为index
            vocab_size: vocabulary的长度（一共有多少个词）
            embedding_dim: 词向量的维度
            hidden_dim：隐藏层的维度
            embedding_mat：预训练的词向量矩阵
    """

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embedding_mat=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # 注：此训练的词向量为一个参数 会反向传播
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if embedding_mat is not None:
            # print(embedding_mat)
            self.word_embeds.load_state_dict({'weight': torch.tensor(embedding_mat)})

        # num_layers —— 叠在一起的lstm层数
        # hidden_size —— 隐状态的维度（注：可以与x维度不同！）
        # LSTM —— 隐状态维度 视为C_0 与 H_0相加 又因为这两个相等 所以除以二
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # 将隐状态映射到targetSize 仅线性加权
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        """
            注意：为从j到i的转换
        """
        # transition矩阵通过反向传播学习
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 不可能从x转换到start
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # 不可能从stop转换到其他tag

        # 初始的隐状态H_0+C_0
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    """
        功能：相较于forward的维特比算法，这个简化版本并未计算bptr
        输入：feats: BLSTM对于每个tag的输出
        输出：最优路径的score
    """

    def _forward_alg(self, feats):
        # 初始化score为-inf
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 起始路径的score为0
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            # 直接使用矩阵运算一步完成
            s = feat.view(-1, 1).expand(-1, self.tagset_size) + forward_var.expand(self.tagset_size,
                                                                                   -1) + self.transitions
            # 直接加s也可以 但是会溢出 因此减去max_score后进行log再重新加上
            max_score, idx = torch.max(s, 1)
            max_score_broadcast = max_score.view(-1, 1).expand(-1, s.size()[1])
            # 相当于log_sum_exp(s)
            e = max_score + torch.log(torch.sum(torch.exp(s - max_score_broadcast), dim=1))
            # 使用view来保证维度
            forward_var = e.view(1, -1)
            # 未优化的代码如下：
            # alphas_t = []  # The forward tensors at this timestep
            # # tagset_size-> 目标分类标签的维度
            # for next_tag in range(self.tagset_size):
            #     # broadcast the emission score: it is the same regardless of
            #     # the previous tag
            #     emit_score = feat[next_tag].view(
            #          1, -1).expand(1, self.tagset_size)
            #     # the ith entry of trans_score is the score of transitioning to
            #     # next_tag from i
            #     trans_score = self.transitions[next_tag].view(1, -1)
            #     # The ith entry of next_tag_var is the value for the
            #     # edge (i -> next_tag) before we do log-sum-exp
            #     next_tag_var = forward_var + trans_score  + emit_score
            #     # The forward variable for this tag is log-sum-exp of all the
            #     # scores.
            #     alphas_t.append(log_sum_exp(next_tag_var).view(1))  # 存储从本时间点到此tag的最优score的损失
            # # # 将列表转换为1 * tag_size的行向量
            # forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    """
        输入：sentence
        输出：状态函数——emission score of the tags
    """

    def _get_lstm_features(self, sentence):
        # 初始化隐状态h_0，c_0
        self.hidden = self.init_hidden()
        # 使用词向量输入
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # output is of shape (seq_len, batch, num_directions * hidden_size)
        # batch = 1 重新扁平化
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    """
        功能：利用已知的tag + transitions矩阵计算真实得分
    """

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # 注：此tags非彼tags tags[i+1] = 原tag[i] 所以tags[i+1]对应feats[i]
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]  # 在对应位置上成为相应标签的score
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]  # 从最后一个tag转换到STOP_TAG
        return score

    # CRF
    # 在log space，乘法的似然变成了加
    def _viterbi_decode(self, feats):
        backpointers = []

        # 初始化score为-inf
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # 存储最优的路径score
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # 不需要加上本位置的emission score，因为比较大小都一样
                next_tag_var = forward_var + self.transitions[next_tag]  # 到下一个tag的score 向量加法 一次即可算出所有值
                best_tag_id = argmax(next_tag_var)  # 寻找到下一个确定的tag 最优的路径（确定到下一个确定的tag 本列最优的tag）
                bptrs_t.append(best_tag_id)  # 存储最优路径
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  # 存储到达此tag的最优路径值
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 最后再加上emission score（出于计算上的考虑——优化计算流程）
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # 加上这一列的bptr（是从上一列的哪个tag转换而来的）

        # Transition to STOP_TAG
        # 确定最后一列的最优tag
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]  # 最终的score

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]  # best_tag_id——为到stop_TAG的前一个tag
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check——保证start为pop掉的tag
        best_path.reverse()  # reverse即可得到最终path
        return path_score, best_path

    # 负对数似然
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq



# 功能：训练
def train(HIDDEN_DIM=200, dir_path="../data/model/blstm-crf/", mod=BiLSTM_CRF):
    trPath = "../data/corpus/msr_training.utf8"
    testPath = "../data/corpus/msr_test_gold.utf8"

    EMBEDDING_DIM = 300

    # fetch the training data and the test data
    x_raw, y_raw = readCorpus(trPath)
    x_test, y_test = readCorpus(testPath)
    # 获取在数据集上的词汇编码
    word_to_ix = getWord2Ix(x_raw)
    # fetch the training data 、test data and validate data
    x_tr, x_val, y_tr, y_val = train_test_split(x_raw, y_raw, test_size=0.1, random_state=1)
    training_data = pack(x_tr, y_tr)
    val_data = pack(x_val, y_val)
    test_data = pack(x_test, y_test)
    #
    # # 获取在数据集上的词汇编码
    # word_to_ix = getWord2Ix(raw_data)

    # 获得本数据集对应的词向量矩阵
    embed_mat_path = "../data/word2vec/embed_mat.npy"
    emb_weight = getEmbedding(word_to_ix, embed_mat_path)

    # 使用cuda加速运算
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mod(len(word_to_ix), SBME_tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, embedding_mat=emb_weight)

    # weight_decay: L2正则化权重的lambda、lr：学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.86)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=True)
    # path
    model_path = dir_path + "model.pt"
    best_f1_path = dir_path + "bestF1.npy"
    best_model_path = dir_path + "bestModel.pt"
    # 加载之前训练的模型
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    # 加载之前最优的验证f1值
    best_f1 = 0.5  # baseline
    if os.path.isfile(best_f1_path):
        best_f1 = np.load(best_f1_path)
        print("Previous best f1: %f" % best_f1)

    iter_times = 100
    # tol_count: 判断迭代是否应该终止
    tol_count = 0
    last_f1 = best_f1
    for epoch in range(iter_times):
        print("%d/%d" % (epoch, iter_times))
        # tags是sentence每个词对应的标记 是一个向量
        # print(epoch)
        torch.save(model.state_dict(), model_path)
        # 随机排序——随机batch
        np.random.shuffle(training_data)
        for sentence, tags in tqdm(training_data, leave=False):
            if len(sentence) > 0:
                # Step 1. 清零积累的梯度
                model.zero_grad()

                # Step 2. 将输入转化为tensor
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.tensor([SBME_tag_to_ix[t] for t in tags], dtype=torch.long)

                # Step 3. 前向计算负对数似然——损失函数.
                loss = model.neg_log_likelihood(sentence_in, targets)

                # Step 4. 反向传播
                loss.backward()
                optimizer.step()
        # endfor
        # 早停法，保存在验证集上F1率最高的模型
        pre_tags = getPrediction(x_val, model, word_to_ix, SBME_ix_to_tag)
        p, r, f1 = SBMS_Validate(y_val, pre_tags)
        if f1 > best_f1:
            best_f1 = f1
            np.save(best_f1_path, f1)
            torch.save(model.state_dict(), best_model_path)
            print("best iter: %d" % epoch)
            print("best f1: %f" % f1)
            tol_count = 0
        else:  # best_f1 > f1:
            tol_count += 1
            if tol_count >= 20:
                print("20 continuous iteration with no best_f1_score increase")
                break
        # else:
        #    tol_count = 0
        # endif
        # last_f1 = f1
    # endfor
    # 存储最后一次的训练
    # torch.save(model.state_dict(), model_path)

    # 加载最优模型
    model.load_state_dict(torch.load(best_model_path))

    # 在测试集上进行测试、打印指标P R F值
    print("Test:")
    pre_tags = getPrediction(x_test, model, word_to_ix, SBME_ix_to_tag)
    SBMS_Validate(test_data, pre_tags)

    # 最终在验证集上的指标
    print("\nValidate:")
    pre_tags = getPrediction(x_val, model, word_to_ix, SBME_ix_to_tag)
    SBMS_Validate(test_data, pre_tags)

    # 最终在训练集上的指标：
    print("\nTrain:")
    pre_tags = getPrediction(x_test, model, word_to_ix, SBME_ix_to_tag)
    SBMS_Validate(test_data, pre_tags)


if __name__ == '__main__':
    # train(50, "../data/model/hid50/")
    # train(100, "../data/model/hid100/")
    train()
