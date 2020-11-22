from utils import *
from sklearn.model_selection import train_test_split
import torch.nn as nn
from TorchCRF import CRF
import torch.optim as optim
from tqdm import trange
from data import *

torch.manual_seed(1)

# TODO:将batch的模型编写
# TODO：LSTM的batch流程

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
        self.tagset_size = len(tag_to_ix) - 2   # 减去start + stop

        # 注：此训练的词向量为一个参数 会反向传播
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if embedding_mat is not None:
            # print(embedding_mat)
            self.word_embeds.load_state_dict({'weight': torch.tensor(embedding_mat)})

        # num_layers —— 叠在一起的lstm层数
        # hidden_size —— 隐状态的维度（注：可以与x维度不同！）
        # LSTM —— 隐状态维度 视为C_0 与 H_0相加 又因为这两个相等 所以除以二
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # 将隐状态映射到targetSize 仅线性加权
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF的实现自动添加start 与 end
        self.crf = CRF(len(tag_to_ix) - 2, use_gpu=False)

    #     # 初始的隐状态H_0+C_0
    #     self.hidden = self.init_hidden()
    #
    # def init_hidden(self):
    #     return (torch.randn(2, 1, self.hidden_dim // 2),
    #             torch.randn(2, 1, self.hidden_dim // 2))


    """
        输入：已batch的sentence
        输出：状态函数——emission score of the tags
    """
    def _get_lstm_features(self, sentences):
        # batch_size = len(sentences)
        # 初始化隐状态h_0，c_0
        # self.hidden = self.init_hidden()
        # 使用词向量输入
        embeds = self.word_embeds(sentences)
        lstm_out, _ = self.lstm(embeds)
        # output is of shape (seq_len, batch, num_directions * hidden_size)
        # batch = 1 重新扁平化
        # lstm_out = lstm_out.view(len(sentences), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    # 负对数似然
    def neg_log_likelihood(self, sentences, tags, mask):
        emissions = self._get_lstm_features(sentences)
        losses = self.crf.forward(emissions, tags, mask)
        loss = torch.sum(losses) / len(sentences)
        # 负对数似然
        return -loss

    def forward(self, sentences, masks):

        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentences)

        # Find the best path, given the features.
        tag_seq = self.crf.viterbi_decode(lstm_feats, masks)
        return tag_seq


# 功能：训练
def train(HIDDEN_DIM=200, dir_path="../data/model/batch-blstm-crf/", mod=BiLSTM_CRF):
    trPath = "../data/corpus/msr_training.utf8"
    testPath = "../data/corpus/msr_test_gold.utf8"

    EMBEDDING_DIM = 300

    # fetch the training data and the test data
    rawSet = FenCiDataset(trPath)
    ratio = 0.1
    trSet, valSet = rawSet.split(1-ratio, ratio)
    testSet = FenCiDataset(testPath)
    # 获取在数据集上的词汇编码
    word_to_ix = getWord2Ix(rawSet.x_data)
    # word to idx
    vocab = word2Idx(getWord2Ix(trSet.x_data))
    batchSize = 200
    trLoader = FenCiDataLoader(trSet, vocab, batch_size=batchSize)
    valLoader = FenCiDataLoader(valSet, vocab, batch_size=batchSize)
    testLoader = FenCiDataLoader(testSet, vocab, batch_size=batchSize)

    # # 获取在数据集上的词汇编码
    # word_to_ix = getWord2Ix(raw_data)

    # 获得本数据集对应的词向量矩阵
    embed_mat_path = "../data/word2vec/embed_mat.npy"
    emb_weight = getEmbedding(word_to_ix, embed_mat_path)

    # 使用cuda加速运算
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mod(len(word_to_ix), SBME_tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, embedding_mat=emb_weight)
    # model.to(device)
    # weight_decay: L2正则化权重的lambda、lr：学习率
    optimizer = optim.SGD(model.parameters(), lr=0.002, weight_decay=1e-4, momentum=0.9)
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

    # 在测试集上进行测试、打印指标P R F值
    print("Test:")
    pre_tags = getPrediction(testLoader, model, SBME_ix_to_tag)
    SBMS_Validate(testSet.y_data, pre_tags)

    iter_times = 100
    # tol_count: 判断迭代是否应该终止
    tol_count = 0
    # 每个epoch测试的次数
    val_count = 5
    for epoch in range(iter_times):
        print("\nEpoch: %d/%d" % (epoch, iter_times))
        # tags是sentence每个词对应的标记 是一个向量
        # print(epoch)
        # shuffle 数据集
        trLoader.shuffle()
        for i, data in enumerate(tqdm(trLoader)):
            # Step 0. load data
            sentences, tags, masks = data
            # Step 1. 清零积累的梯度
            model.zero_grad()

            # Step 2. 前向计算负对数似然——损失函数.
            loss = model.neg_log_likelihood(sentences, tags, masks)

            # Step 3. 反向传播
            loss.backward()
            optimizer.step()

            # Step 4. 验证训练效果
            # 早停法，保存在验证集上F1率最高的模型
            if ((i + 1) % (len(trLoader) // val_count)) == 0:
                # 保存训练进度
                torch.save(model.state_dict(), model_path)
                print("\nModel saved.")
                # 验证
                pre_tags = getPrediction(valLoader, model, SBME_ix_to_tag)
                f1 = SBMS_Validate(valSet.y_data, pre_tags)
                if f1 > best_f1:
                    best_f1 = f1
                    np.save(best_f1_path, f1)
                    torch.save(model.state_dict(), best_model_path)
                    # print("best iter: %d" % epoch)
                    print("New best f1: %f" % f1)
                    tol_count = 0
                else:  # best_f1 > f1:
                    tol_count += 1
                    if tol_count >= 10:
                        print("10 continuous validation with no best_f1_score increase")
                        break
            # endfor
        # endfor

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
    pre_tags = getPrediction(testLoader, model, SBME_ix_to_tag)
    SBMS_Validate(testSet.y_data, pre_tags)

    # 最终在验证集上的指标
    print("\nValidate:")
    pre_tags = getPrediction(valLoader, model, SBME_ix_to_tag)
    SBMS_Validate(valSet.y_data, pre_tags)

    # 最终在训练集上的指标：
    print("\nTrain:")
    pre_tags = getPrediction(trLoader, model, SBME_ix_to_tag)
    SBMS_Validate(trSet.y_data, pre_tags)


if __name__ == "__main__":
    train()
