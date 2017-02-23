# coding: utf-8
"""
Sequence to Sequenceのchainer実装

Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le.,
"Sequence to sequence learning with neural networks.",
Advances in neural information processing systems. 2014.
"""

import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import datetime
from filer2 import Filer
import glob
import random


class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Encoder, self).__init__(
            xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        """

        :param x: one-hotな単語
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        e = functions.tanh(self.xe(x))
        return functions.lstm(c, self.eh(e) + self.hh(h))

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Decoder, self).__init__(
            ye = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size),
            he = links.Linear(hidden_size, embed_size),
            ey = links.Linear(embed_size, vocab_size)
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

class Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, flag_gpu=True):
        """
        Seq2Seqの初期化
        :param vocab_size: 語彙サイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 中間ベクトルのサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Seq2Seq, self).__init__(
            # Encoderのインスタンス化
            encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # Decoderのインスタンス化
            decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # GPUで計算する場合はcupyをCPUで計算する場合はnumpyを使う
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

    def encode(self, words):
        """
        Encoderを計算する部分
        :param words: 単語が記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        # エンコーダーに単語を順番に読み込ませる
        for w in words:
            c, h = self.encoder(w, c, h)

        # 計算した中間ベクトルをデコーダーに引き継ぐためにインスタンス変数にする
        self.h = h
        # 内部メモリは引き継がないので、初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        デコーダーを計算する部分
        :param w: 単語
        :return: 単語数サイズのベクトルを出力する
        """
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t

    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        :return:
        """
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        self.zerograds()


def forward(enc_words, dec_words, model, ARR):
    """
    順伝播の計算を行う関数
    :param enc_words: 発話文の単語を記録したリスト
    :param dec_words: 応答文の単語を記録したリスト
    :param model: Seq2Seqのインスタンス
    :param ARR: cuda.cupyかnumpyか
    :return: 計算した損失の合計
    """
    # バッチサイズを記録
    batch_size = len(enc_words[0])
    # model内に保存されている勾配をリセット
    model.reset()
    # 発話リスト内の単語を、chainerの型であるVariable型に変更
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    # エンコードの計算 ①
    model.encode(enc_words)
    # 損失の初期化
    loss = Variable(ARR.zeros((), dtype='float32'))
    # <eos>をデコーダーに読み込ませる ②
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

def forward_test(enc_words, model, ARR):
    ret = []
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0], dtype='int32'))
    counter = 0
    while counter < 50:
        y = model.decode(t)
        label = y.data.argmax()
        ret.append(label)
        t = Variable(ARR.array([label], dtype='int32'))
        counter += 1
        if label == 1:
            counter = 50
    return ret

# trainの関数
def make_minibatch(minibatch):
    # enc_wordsの作成
    enc_words = [row[0] for row in minibatch]
    enc_max = np.max([len(row) for row in enc_words])
    enc_words = np.array([[-1]*(enc_max - len(row)) + row for row in enc_words], dtype='int32')
    enc_words = enc_words.T

    # dec_wordsの作成
    dec_words = [row[1] for row in minibatch]
    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([row + [-1]*(dec_max - len(row)) for row in dec_words], dtype='int32')
    dec_words = dec_words.T
    return enc_words, dec_words

def train():
    # 辞書の読み込み
    word_to_id = Filer.readdump(DICT_PATH)
    # 語彙数
    vocab_size = len(word_to_id)
    # モデルのインスタンス化
    model = Seq2Seq(vocab_size=vocab_size,
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=BATCH_SIZE,
                    flag_gpu=FLAG_GPU)
    model.reset()
    # GPUのセット
    if FLAG_GPU:
        ARR = cuda.cupy
        cuda.get_device(0).use()
        model.to_gpu(0)
    else:
        ARR = np

    # 学習開始
    for epoch in range(EPOCH_NUM):
        # ファイルのパスの取得
        filepath = glob.glob(INPUT_PATH)
        # エポックごとにoptimizerの初期化
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))
        for path in filepath:
            # ファイルの読み込み
            data = Filer.readdump(path)
            print (path)
            random.shuffle(data)
            for num in range(len(data)//BATCH_SIZE):
                minibatch = data[num*BATCH_SIZE: (num+1)*BATCH_SIZE]
                # 読み込み用のデータ作成
                enc_words, dec_words = make_minibatch(minibatch)
                # modelのリセット
                model.reset()
                # 順伝播
                total_loss = forward(enc_words=enc_words,
                                     dec_words=dec_words,
                                     model=model,
                                     ARR=ARR)
                # 学習
                total_loss.backward()
                opt.update()
                opt.zero_grads()
                # print (datetime.datetime.now())
        print ('Epoch %s 終了' % (epoch+1))
        SlackBot.post_message(username=username,
                          text='Epoch %s 終了' % (epoch+1),
                          icon_emoji=':thumbsup:')
        outputpath = OUTPUT_PATH%(EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, epoch+1)
        serializers.save_hdf5(outputpath, model)


INPUT_PATH = '../file/train/ntcir/que_ans_id.dump'
OUTPUT_PATH = '../file/model/seq2seq/ntcir/EMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.weights'
DICT_PATH = '../file/train/ntcir/word_to_id.dump'
FLAG_GPU = True

EMBED_SIZE = 300
HIDDEN_SIZE = 150
BATCH_SIZE = 40
EPOCH_NUM = 30

username='Seq2Seq'

print ('開始: ', datetime.datetime.now())
try:
    train()
except:
    SlackBot.post_message(username=username,
                          text='エラー、下手くそ！！',
                          icon_emoji=':troll:')
    raise

print ('終了: ', datetime.datetime.now())