# coding: utf-8
"""
Vinyals, Oriol, et al.
"Show and tell: A neural image caption generator."
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition. 2015.
"""

import numpy as np

from chainer import functions, links, Chain, Variable, cuda

class AlexNet(Chain):
    insize = 224
    def __init__(self, hidden_size, train=True):
        super(AlexNet, self).__init__(
            conv1=links.Convolution2D(3,  96, 11, stride=4),
            conv2=links.Convolution2D(96, 256,  5, pad=2),
            conv3=links.Convolution2D(256, 384,  3, pad=1),
            conv4=links.Convolution2D(384, 384,  3, pad=1),
            conv5=links.Convolution2D(384, 256,  3, pad=1),
            fc6=links.Linear(9216, 4096),
            fc7=links.Linear(4096, 1000),
            fc8=links.Linear(1000, hidden_size),
        )
        self.train = train

    def __call__(self, x):
        h = functions.max_pooling_2d(functions.local_response_normalization(
            functions.relu(self.conv1(x))), 3, stride=2)
        h = functions.max_pooling_2d(functions.local_response_normalization(
            functions.relu(self.conv2(h))), 3, stride=2)
        h = functions.relu(self.conv3(h))
        h = functions.relu(self.conv4(h))
        h = functions.max_pooling_2d(functions.relu(self.conv5(h)), 3, stride=2)
        print ()
        h = functions.dropout(functions.relu(self.fc6(h)), train=self.train)
        h = functions.dropout(functions.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h


class Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(Decoder, self).__init__(
            ye=links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=links.Linear(embed_size, 4 * hidden_size),
            hh=links.Linear(hidden_size, 4 * hidden_size),
            he=links.Linear(hidden_size, embed_size),
            ey=links.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        """

        :param y: one-hotな単語
        :param c: 内部メモリ
        :param h: 隠れそう
        :return: 予測単語、次の内部メモリ、次の隠れ層
        """
        e = functions.tanh(self.ye(y))
        c, h = functions.lstm(c, self.eh(e) + self.hh(h))
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h


class CaptionGenerator(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, train=True, gpu=True):
        super(CaptionGenerator, self).__init__(
            cnn=AlexNet(hidden_size=hidden_size,
                        train=train),
            decoder = Decoder(vocab_size=vocab_size,
                              embed_size=embed_size,
                              hidden_size=hidden_size)
        )
        if gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

    def encode(self, x):
        self.h = self.cnn(x)

    def decode(self, y):
        t, self.c, self.h = self.decoder(y, self.c, self.h)
        return t

    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        :return:
        """
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        self.zerograds()


def forward(image, dec_words, model, ARR):
    batch_size = len(image)
    # model内に保存されている勾配をリセット
    model.reset()
    # 発話リスト内の単語を、chainerの型であるVariable型に変更
    image = Variable(ARR.array(image, dtype='float32'))
    # エンコードの計算 ①
    model.encode(image)
    # 損失の初期化
    loss = Variable(ARR.zeros((), dtype='float32'))
    # <bos>をデコーダーに読み込ませる ②
    t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
    # デコーダーの計算
    for w in dec_words:
        # 1単語ずつをデコードする ③
        y = model.decode(t)
        # 正解単語をVariable型に変換
        t = Variable(ARR.array(w, dtype='int32'))
        # 正解単語と予測単語を照らし合わせて損失を計算 ④
        loss += functions.softmax_cross_entropy(y, t)
    return loss


def forward_test(image, model, ARR):
    model.reset()
    image = Variable(ARR.array(image, dtype='float32'))
    model.encode(image)
    t = Variable(ARR.array([0], dtype='int32'))
    ret = [0]
    counter = 0
    while True:
        y = model.decode(t)
        w = y.data[0].argmax()
        ret.append(w)
        t = Variable(ARR.array([w], dtype='int32'))
        counter += 1
        if w == 1 or counter > 50:
            break
    return ret
