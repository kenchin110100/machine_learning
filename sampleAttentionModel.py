# coding: utf-8
"""
Attention Modelで文書の分類を行う
"""
import sys
import numpy
from argparse import ArgumentParser
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import collections
import numpy as np
import random
from filer2.filer2 import Filer

# コーパスをid化するための関数
def make_corpus(list_corpus, lower_freq=1):
    """
    list_corpus: 形態素解析された文がリストとして格納されたリスト
    lower_freq: 少ない頻度の単語を<unk>として処理する
    return: list_corpus_id id化されたコーパス
          : dict_word_id: 単語とそのidが記録された辞書
    """
    # 全ての単語を一旦リストに格納して、出現頻度を計算する
    list_all_word = [word for row in list_corpus for word in row]
    list_word_freq = collections.Counter(list_all_word)
    # 頻度がlower_freq以下の単語を<unk>に変える、その際、先頭記号<s>, 終末記号</s>を追加
    list_unk_word = [word for word, freq in list_word_freq.items() if freq <= lower_freq]
    list_corpus_rev = [['<s>']+['<unk>' if word in list_unk_word else word for word in row]+['</s>'] for row in list_corpus]
    # 辞書を作成
    dict_word_id = {'<s>': 0, '</s>':1, '<unk>':2}
    counter = 3
    for row in list_corpus_rev:
        for word in row:
            if word not in dict_word_id:
                dict_word_id[word] = counter
                counter += 1
    # 作成した辞書を用いて、corpus中の単語をidに変更
    list_corpus_id = [[dict_word_id[word] for word in row] for row in list_corpus_rev]
    # 作成したコーパスと辞書を返す
    return list_corpus_id, dict_word_id

# 単語idをembedするクラス、tanhを噛ませているのがポイント
class SrcEmbed(Chain):
    def __init__(self, vocab_size, embed_size):
        super(SrcEmbed, self).__init__(
            xe = links.EmbedID(vocab_size, embed_size),
        )

    def __call__(self, x):
        return functions.tanh(self.xe(x))

# lstmのエンコーダ
class LSTMEncoder(Chain):
    def __init__(self, embed_size, hidden_size):
        super(LSTMEncoder, self).__init__(
            lstm = links.LSTM(embed_size, hidden_size),
        )
    def reset(self):
        self.zerograds()
    def __call__(self, x):
        h = self.lstm(x)
        return h

# Attentionするクラス、考慮の余地あり、いらない線形結合がある気がする
# self.emphaの中にattetionの割合を記録している、可視化するときにはこれを使用する
class Attention(Chain):
    def __init__(self, hidden_size):
        super(Attention, self).__init__(
            pw = links.Linear(hidden_size, hidden_size),
            we = links.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, a_list):
        e_list = []
        self.empha = []
        sum_e = Variable(np.array([[0]], dtype='float32'))
        for a in a_list:
            w = functions.tanh(self.pw(a))
            e = functions.exp(self.we(w))
            e_list.append(e)
            sum_e += e

        ZEROS = Variable(np.zeros((1, self.hidden_size), dtype='float32'))
        aa = ZEROS
        for a, e in zip(a_list, e_list):
            e /= sum_e
            self.empha.append(e)
            aa += a * functions.broadcast_to(e, (1, self.hidden_size))
        #aa += functions.reshape(functions.batch_matmul(a, e), (batch_size, self.hidden_size))
        return aa

# メインのクラス
class AttentionLM(Chain):
    def __init__(self, embed_size, hidden_size, vocab_size, label_size):
        super(AttentionLM, self).__init__(
            emb = SrcEmbed(vocab_size, embed_size),
            enc = LSTMEncoder(embed_size, hidden_size),
            att = Attention(hidden_size),
            outae = links.Linear(hidden_size, hidden_size),
            outey = links.Linear(hidden_size, label_size),
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.label_size = label_size

    def reset(self):
        self.zerograds()
        self.x_list = []
        self.h_list = []

    def embed(self, x):
        self.x_list.append(self.emb(x))

    def encode(self):
        for x in self.x_list:
            self.h = self.enc(x)
            self.h_list.append(self.h)

    def decode(self):
        aa = self.att(self.h_list)
        y = functions.tanh(self.outae(aa))
        return self.outey(y)

# 順伝播の計算
def forward(list_corpus_one, label, model):
    L = len(list_corpus_one)

    opt.zero_grads()

    for one in list_corpus_one:
        model.embed(Variable(np.array([one], dtype='int32')))
    s_t = Variable(np.array([label], dtype='int32'))
    model.encode()
    s_y = model.decode()
    loss_i = functions.softmax_cross_entropy(s_y, s_t)

    return loss_i

# メインの関数
def main():
    # list_corpus_id, dict_word_id = make_corpus(list_corpus, lower_freq=1)
    # list_corpus_id_y = [[row, y] for row, y in zip(list_corpus_id, list_y)]

    # モデルのインスタンス化
    model = AttentionLM(embed_size=embed_size,
                        hidden_size=hidden_size,
                        vocab_size=len(dict_word_id),
                        label_size=label_size)
    # パラメータの初期化
    model.reset()

    # エポック分だけ回す
    for i in range(epoch):
        # 無難にAdamを使用
        opt = optimizers.Adam()
        opt.setup(model)
        # clipはよく理解していない・・・
        opt.add_hook(optimizer.GradientClipping(5))
        # コーパスのシャッフル
        random.shuffle(list_corpus_id_y)
        # 1文ごとに計算して、逆伝播
        for list_corpus_one, y in list_corpus_id_y:
            # 順伝播
            loss = forward(list_corpus_one=list_corpus_one,
                           label=y,
                           model=model)
            # 逆伝播
            loss.backward()
            # 勾配を引く
            opt.update()
            # パラメータをリセット
            model.reset()
        print 'Epoch: ', i
        # エポックごとに保存
        # serializers.save_hdf5('%s.weights'%str(epoch), model)


if __name__ == '__main__':
    # エポック数
    epoch = 1
    # embedのサイズ
    embed_size = 300
    # 隠れ層のサイズ
    hidden_size = 50
    # 分類するラベルのサイズ
    label_size = 3
    # 全体コーパスをidに変換
    list_corpus_id, dict_word_id =make_corpus(list_corpus_id, lower_freq=1)
    # 辞書の保存
    Filer.writedump(dict_word_id, './dict_word_id.dump')
    # 計算
    main()


