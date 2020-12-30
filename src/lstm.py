import torch.nn as nn
from blstm import train
from data import *

torch.manual_seed(1)
# 需要padding为0 idx
SBME_tag_to_ix = {PADDING: 0, "S": 1, "B": 2, "M": 3, "E": 4,  START_TAG: 5, STOP_TAG: 6}

class LSTM(nn.Module):
    """
        输入：tag_to_ix：字典：将所有的标签转换为index
            vocab_size: vocabulary的长度（一共有多少个词）
            embedding_dim: 词向量的维度
            hidden_dim：隐藏层的维度
            embedding_mat：预训练的词向量矩阵
    """

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embedding_mat=None):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix) - 2  # 减去start + stop

        # 注：此训练的词向量为一个参数 会反向传播
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if embedding_mat is not None:
            # print(embedding_mat)
            self.word_embeds.load_state_dict({'weight': torch.tensor(embedding_mat)})

        # num_layers —— 叠在一起的lstm层数
        # hidden_size —— 隐状态的维度（注：可以与x维度不同！）
        # LSTM —— 隐状态维度 视为C_0 与 H_0相加 又因为这两个相等 所以除以二
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=1, bidirectional=False)

        # Maps the output of the LSTM into tag space.
        # 将隐状态映射到targetSize 仅线性加权
        self.hidden2tag = nn.Linear(hidden_dim // 2, self.tagset_size)


    """
        输入：已batch的sentence
        输出：状态函数——emission score of the tags
    """
    def get_lstm_features(self, sentences, masks, pad=False):
        # 使用词向量输入
        embeds = self.word_embeds(sentences)
        if pad:     # 使用pack_padded_sequence
            # get input length
            input_length = torch.sum(masks, dim=1)
            # padding 注意到此函数只能用cpu的length
            packed = torch.nn.utils.rnn.\
                pack_padded_sequence(embeds, input_length.to(torch.device("cpu")), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            # Unpack
            lstm_out = torch.nn.utils.rnn.\
                pad_packed_sequence(lstm_out, batch_first=True, padding_value=0)
            lstm_out = lstm_out[0]  # [1]为长度
        else:
            lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward_alg(self, sentences, masks):

        # Get the emission scores from the BiLSTM
        lstm_feats = self.get_lstm_features(sentences, masks)
        return lstm_feats

    def forward(self, sentences, masks):
        # Get the emission scores from the BiLSTM
        lstm_feats = self.get_lstm_features(sentences, masks)
        _, predicted = torch.max(lstm_feats, 2)
        p = predicted.to(torch.device("cpu"))
        tags = p.numpy()
        result = []
        for i, t in enumerate(tags):
            num = torch.sum(masks[i]).item()
            result.append(tags[i][:num])
        return result


if __name__ == "__main__":
    train(b_size=200, use_cuda=True, mod=LSTM, dir_path="../data/model/batch-lstm/")
