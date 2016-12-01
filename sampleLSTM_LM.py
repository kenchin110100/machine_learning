# coding: utf-8
"""
chainerのLSTMで言語モデルを学習する
"""
from filer2.filer2 import Filer

import argparse
import math
import sys
import time
import random
import glob

import numpy as np
import six

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from chainer import cuda

import collections


# 順伝播のLSTMモデル
# LSTMは２層

class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units, train=True):
        """
        n_vocab: 総語彙数
        n_unit: 隠れ層の数
        """
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        # パラメータの初期化
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        # 順伝播のオーバーライド
        self.train = train
    # 隠れ層の初期化
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
    # コールされた場合、順伝播させていき、最終的に、順伝播の結果を返す
    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y

# 総単語数
p = 108047
# 隠れ層のunit数
n_units = 650
# バッチサイズ
num_batch = 100
# エポック数
num_epoch = 30
# クリップサイズ
grad_clip = 5
# モデルを書き出すタイミング
total_counter = 10000

# コーパスのパスの読み込み
list_path = glob.glob('./files/corpus/reshape/*.csv')

# コーパスの読み込み
list_master = [Filer.readcsv(path, option='r') for path in list_path]
print ('コーパスの読み込み終了: ', time.strftime("%Y/%m/%d %H:%M:%S"))

lstm = RNNForLM(p , n_units)
model = L.Classifier(lstm)
model.compute_accuracy = False
# 勾配法の選択
optimizer = optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

print ('モデルの初期化終了: ', time.strftime("%Y/%m/%d %H:%M:%S"))

# GPUの準備
# deviceで数字を指定する必要がある
cuda.get_device(0).use()
model.to_gpu(0)
xp = cuda.cupy

# ミニバッチの作成に使用する関数
def make_minibatch(list_len_sentence, num_batch):
    list_return = []
    for row in list_len_sentence:
        cycle = len(row) // num_batch
        random.shuffle(row)
        for i in range(cycle):
            list_return.append(np.array(row[i*num_batch:(i+1)*num_batch], dtype='int32').T)
    random.shuffle(list_return)
    return list_return

# 実際の学習部分
print ('学習開始: ', time.strftime("%Y/%m/%d %H:%M:%S"))

for epoch in range(1, num_epoch+1):
    list_batch = make_minibatch(list_master, num_batch)
    print ('ミニバッチの作成終了: ', time.strftime("%Y/%m/%d %H:%M:%S"))
    counter = 1
    for batch in list_batch:
        accum_loss = 0
        lstm.reset_state()
        for i in range(len(batch)-1):
            x = chainer.Variable(xp.array(batch[i], dtype='int32'))
            t = chainer.Variable(xp.array(batch[i+1], dtype='int32'))
            accum_loss += model(x, t)
        model.zerograds()
        accum_loss.backward()
        optimizer.update()
        counter += 1
        if counter % total_counter == 0:
            serializers.save_npz('./files/model/rnnlm.model_%s'%(str(epoch)+str(counter/total_counter)), model)
            serializers.save_npz('./files/model/rnnlm.state_%s'%(str(epoch)+str(counter/total_counter)), optimizer)
            print ('モデルを書き出しました', time.strftime("%Y/%m/%d %H:%M:%S"))
    print ('epoch: ', epoch, ' 終了')
    serializers.save_npz('./files/model/rnnlm.model_%s'%(str(epoch)), model)
    serializers.save_npz('./files/model/rnnlm.state_%s'%(str(epoch)), optimizer)
    print (time.strftime("%Y/%m/%d %H:%M:%S"))
