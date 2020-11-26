import copy

import torch
import numpy as np
import torch.utils.data as D
from preprocess import *


class FenCiDataset(D.Dataset):
    def __init__(self, path):
        self.x_data, self.y_data = readCorpus(path)

        assert len(self.x_data) == len(self.y_data), \
            "not all words matches tags"

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.x_data)

    def shuffle(self):
        idx = np.arange(len(self))
        idx = np.random.permutation(idx)  # 复制并重新排序
        self.x_data = [self.x_data[i] for i in idx]
        self.y_data = [self.y_data[i] for i in idx]

    def split(self, *ratios, shuffle=False):
        idx = np.arange(len(self))

        if shuffle:
            idx = np.random.permutation(idx)  # 复制并重新排序

        # 各比例
        ratios = [r / sum(ratios) for r in ratios]
        # 个比例对应的数目
        counts = [int(round(len(self) * r)) for r in ratios]
        # 得到边界
        cum_counts = [sum(counts[:i + 1]) for i in range(len(ratios))]
        # 补上开头
        bounds = [0] + cum_counts

        for i in range(len(bounds) - 1):
            s = copy.copy(self)
            s.x_data = [self.x_data[j] for j in idx[bounds[i]:bounds[i + 1]]]
            s.y_data = [self.y_data[j] for j in idx[bounds[i]:bounds[i + 1]]]

            yield s


class FenCiDataLoader(D.DataLoader):
    def __init__(self, dataset, word2idx: word2Idx, device="cpu", tensor_lens=True, mode="SBME", **kwargs):
        super(FenCiDataLoader, self).__init__(
            dataset=dataset,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.mode = mode
        self.tensor_lens = tensor_lens
        self.word2idx = word2idx
        self.device = device

    def unlexicalize(self, sent):
        return [self.word2idx[w] for w in sent]  # return [self.word2idx[w] for w in sent]

    # 功能：pad x,y 返回masks
    def pad(self, x_raw, y_raw):
        lens = [len(s) for s in x_raw]
        max_len = max(lens)
        assert len(x_raw) == len(y_raw)
        x_pad, y_pad, masks = [], [], []
        for i in range(len(x_raw)):
            padding = "<PAD>"
            assert len(x_raw[i]) == len(y_raw[i])
            l = len(x_raw[i])
            x_pad.append(x_raw[i] + [self.word2idx[padding]] * (max_len - l))
            y_pad.append(y_raw[i] + [padding] * (max_len - l))
            masks.append([1] * l + [0] * (max_len - l))
        return x_pad, y_pad, masks

    def tag_2_ix(self, tags):
        if self.mode == "SBME":
            ret = [[SBME_tag_to_ix[t] for t in ts] for ts in tags]
        elif self.mode == "BI":
            ret = [[BI_tag_to_ix[t] for t in ts] for ts in tags]
        else:
            raise ValueError("Wrong mode.")
        return ret

    # 读取batch的时候处理数据
    def collate_fn(self, batches):
        x_raw = []
        y_raw = []
        for _x, _y in batches:
            x_raw.append(self.unlexicalize(_x))
            y_raw.append(_y)
        x, y, m = self.pad(x_raw, y_raw)
        # 返回batch pad+转为idx后的张量 以及可用于mask的lens
        return torch.LongTensor(x).to(self.device), \
               torch.LongTensor(self.tag_2_ix(y)).to(self.device), \
               torch.ByteTensor(m).to(self.device)

    # shuffle 数据
    def shuffle(self):
        self.dataset.shuffle()


if __name__ == "__main__":
    dset = FenCiDataset("../data/corpus/msr_test_gold.utf8")
    loader = FenCiDataLoader(dset, word2Idx(getWord2Ix(dset.x_data)), batch_size=50)
    ep = 3
    for e in range(ep):
        loader.shuffle()
        for x, y, m in loader:
            print("\nX:")
            print(x)
            print("Y:")
            print(y)
            print("mask:")
            print(m)
            break
