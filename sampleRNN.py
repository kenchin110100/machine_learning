# coding: utf-8
"""
ミニバッチ:１０
計算の工夫あり（SGD改良）
bあり
"""

from chainer import FunctionSet, Variable, optimizer
from chainer.functions import *
from chainer.optimizers import *
from gensim import corpora
import math
import numpy as np
import csv
import glob
import MeCab
import pickle
import random

VOCAB_SIZE = 40136            #単語数
HIDDEN_SIZE = 100             #単語ベクトルのサイズ
BATCH_SIZE = 100              #ミニバッチのサイズ

def readcsv(path):
    f = open(path, 'rb')
    dataReader = csv.reader(f)
    arr = [row for row in dataReader]
    return arr

model1 = FunctionSet(w_xh = EmbedID(VOCAB_SIZE, HIDDEN_SIZE))

model2 = FunctionSet(
  w_hh = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 隠れ層 -> 隠れ層
  w_hy = Linear(HIDDEN_SIZE, VOCAB_SIZE) # 隠れ層 -> 出力層
)

def forward(sentence): # sentenceはstrの配列。MeCabなどの出力を想定。
    h = Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32)) # 隠れ層の初期値
    accum_loss = Variable(np.zeros((), dtype=np.float32)) # 累積損失の初期値
    for word in sentence:
        x = Variable(np.array([word], dtype=np.int32))
        u = model2.w_hy(h)
        accum_loss += softmax_cross_entropy(u, x) # 損失の蓄積
        h = tanh(model1.w_xh(x) + model2.w_hh(h)) # 隠れ層の更新

    return accum_loss         # 結合確率の計算結果を返す
        
def save_pickle(parameter, path):
    with open(path, 'w') as f:
        pickle.dump(parameter, f)
        
class SGD_Embedid(optimizer.Optimizer):

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, arr_embedids):
        self.t += 1
        for p, g, s in self.tuples:
            self.update_one(p, g, s, arr_embedids)

    def update_one(self, param, grad, state, arr_embedids):
            self.update_one_cpu(param, grad, state, arr_embedids)

    def update_one_cpu(self, param, grad, state, arr_embedids):
        param[arr_embedids] -= self.lr * grad[arr_embedids]

    def zero_grads(self, arr_embedids):
        _, g0, _ = self.tuples[0]
        g0[arr_embedids] = 0


list_sentences = readcsv("./files/list_id20151207.csv")
list_sentences = [np.array(row, np.int32) for row in list_sentences]


opt1 = SGD_Embedid() # 確率的勾配法を使用
opt2 = SGD() # 確率的勾配法を使用
opt1.setup(model1) # 学習器の初期化
opt2.setup(model2) # 学習器の初期化
opt1.tuples[0][1].fill(0)
opt2.zero_grads()
random.shuffle(list_sentences)
list_minibatch = []
for i, sentence in enumerate(list_sentences):
    list_minibatch.append(sentence)
    if len(list_minibatch) == BATCH_SIZE:
        accum_loss_total = Variable(np.zeros((), dtype=np.float32)) # 累積損失の初期値
        uniq_sentence = np.zeros((), np.int32)
        for batch_sentence in list_minibatch:
            accum_loss_total += forward(batch_sentence) # 損失の計算
            uniq_sentence = np.append(uniq_sentence, batch_sentence)
        accum_loss_total.backward() # 誤差逆伝播
        opt1.clip_grads(10) # 大きすぎる勾配を抑制
        opt2.clip_grads(10) # 大きすぎる勾配を抑制
        uniq_sentence = np.unique(uniq_sentence)
        opt1.update(uniq_sentence) # パラメータの更新
        opt2.update() # パラメータの更新
        opt1.zero_grads(uniq_sentence) # 勾配の初期化
        opt2.zero_grads() # 勾配の初期化
        list_minibatch = []
    if i % 1000 == 999:
            break
