import torch
from seqeval.metrics import f1_score, precision_score, accuracy_score, recall_score, classification_report
from preprocess import *


# 功能：根据BI tag分词
def generateFenCi(texts, tags):
    # 保证程序的正确性
    assert len(texts) == len(tags)
    FenCis = []
    print("\nGenerating fenci...")
    for sent, tag in zip(texts, tags):
        FenCiDeSent = []  # 存储分词的一行句子
        fenci = ""
        for w, t in zip(sent, tag):
            if t == "I":
                fenci += w
            else:  # B tag
                if fenci != "":  # 非空
                    FenCiDeSent.append(fenci)
                    fenci = ""
                fenci += w
            # endif
        # endfor
        # 加上最后一个词
        if fenci != "":
            FenCiDeSent.append(fenci)
        # 此句子分词完毕
        FenCis.append(FenCiDeSent)
    # endfor
    return FenCis


# 功能：将分词结果写入文件
def writeFenCiResult(texts, tags, _path):
    f = open(_path, 'w', encoding="utf-8")
    fenciRes = generateFenCi(texts, [SBME2BI(t) for t in tags])
    print("Writing FenCi to File...")
    for sents in tqdm(fenciRes):
        s = "  ".join(sents)
        f.write(s + "\n")
    # endfor
    print("Done.")


# 功能：获取在数据集上验证的指标——F1分数的平均值
# 输入：y_true为真实的tags、y_pref为预测的tags
def SBME_Validate(_y_true, _y_pred):
    f1 = 0
    # 对每个句子进行计算F1分数
    for gt, pred in zip(_y_true, _y_pred):
        y_true, y_pred = SBME2BI(gt), SBME2BI(pred)
        f1 += f1_score([y_true], [y_pred])
    f1 /= len(_y_true)
    print("F1 score: %f" % f1)
    return f1


# 功能：通过model获取prediction
def getPrediction(loader, the_model, ix_to_tag):
    pre = []
    with torch.no_grad():
        for sentences, y, mask in tqdm(loader):
            prediction = list(the_model(sentences, mask))
            pre.extend([[ix_to_tag[ix] for ix in pred] for pred in prediction])
            # print(pre)
            # print(y)
        # endfor
    return pre


# 功能：类似zip 但是取出不会消失
def pack(x, y):
    res = []
    assert len(x) == len(y)
    for i in range(len(x)):
        res.append((x[i], y[i]))
    return res


"""
    输入：seq：sentence
        to_ix：sentence对应词的字典 将每个词映射到一个index
    输出：将seq转换为index的tensor
    注：torch.long为整数
"""
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
