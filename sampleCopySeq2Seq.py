# coding: utf-8
"""
CopyNet + Seq2Seqのコード
Cao, Ziqiang, et al.
"Joint Copying and Restricted Generation for Paraphrase."
arXiv preprint arXiv:1611.09235 (2016).
"""
import numpy as np
from chainer import Chain, Variable, cuda, functions, links

from sampleSeq2Sep import LSTM_Encoder
from sampleAttSeq2Seq import Attention, Att_LSTM_Decoder


class Copy_Attention(Attention):

    def __call__(self, fs, bs, h):
        """
        Attentionの計算
        :param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
        :param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
        :param h: Decoderで出力された中間ベクトル
        :return att_f: 順向きのEncoderの中間ベクトルの加重平均
        :return att_b: 逆向きのEncoderの中間ベクトルの加重平均
        :return att: 各中間ベクトルの重み
        """
        # ミニバッチのサイズを記憶
        batch_size = h.data.shape[0]
        # ウェイトを記録するためのリストの初期化
        ws = []
        att = []
        # ウェイトの合計値を計算するための値を初期化
        sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = self.hw(functions.tanh(self.fh(f)+self.bh(b)+self.hh(h)))
            att.append(w)
            # softmax関数を使って正規化する
            w = functions.exp(w)
            # 計算したウェイトを記録
            ws.append(w)
            sum_w += w
        # 出力する加重平均ベクトルの初期化
        att_f = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        att_b = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        for i, (f, b, w) in enumerate(zip(fs, bs, ws)):
            # ウェイトの和が1になるように正規化
            w /= sum_w
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_f += functions.reshape(functions.batch_matmul(f, w), (batch_size, self.hidden_size))
            att_b += functions.reshape(functions.batch_matmul(f, w), (batch_size, self.hidden_size))
        att = functions.concat(att, axis=1)
        return att_f, att_b, att


class Copy_Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, flag_gpu=True):
        super(Copy_Seq2Seq, self).__init__(
            # 順向きのEncoder
            f_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # 逆向きのEncoder
            b_encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            # Attention Model
            attention=Copy_Attention(hidden_size, flag_gpu),
            # Decoder
            decoder=Att_LSTM_Decoder(vocab_size, embed_size, hidden_size),
            # λの重みを計算するためのネットワーク
            predictor=links.Linear(hidden_size, 1)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fs = []
        self.bs = []

    def encode(self, words):
        """
        Encoderの計算
        :param words: 入力で使用する単語記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # 先ずは順向きのEncoderの計算
        for w in words:
            c, h = self.f_encoder(w, c, h)
            # 計算された中間ベクトルを記録
            self.fs.append(h)

        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # 逆向きのEncoderの計算
        for w in reversed(words):
            c, h = self.b_encoder(w, c, h)
            # 計算された中間ベクトルを記録
            self.bs.insert(0, h)

        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))


    def decode(self, w):
        """
        Decoderの計算
        :param w: Decoderで入力する単語
        :return t: 予測単語
        :return att: 各単語のAttentionの重み
        :return lambda_: Copy重視かGenerate重視かを判定するための重み
        """
        # Attention Modelで入力ベクトルを計算
        att_f, att_b, att = self.attention(self.fs, self.bs, self.h)
        # Decoderにベクトルを入力
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        # 計算された中間ベクトルを用いてλの計算
        lambda_ = self.predictor(self.h)
        return t, att, lambda_

    def reset(self):
        """
        インスタンス変数を初期化する
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # Encoderの中間ベクトルを記録するリストの初期化
        self.fs = []
        self.bs = []
        # 勾配の初期化
        self.zerograds()


def forward(enc_words, dec_words, model, ARR):
    """
    forwardの計算をする関数
    :param enc_words: 入力文
    :param dec_words: 出力文
    :param model: モデル
    :param ARR: numpyかcuda.cupyのどちらか
    :return loss: 損失
    """
    # バッチサイズを記録
    batch_size = len(enc_words[0])
    # モデルの中に記録されている勾配のリセット
    model.reset()
    # 入力文の中で使用されている単語をチェックするためのリストを用意
    enc_key = enc_words.T
    # Encoderに入力する文をVariable型に変更する
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    # Encoderの計算
    model.encode(enc_words)
    # 損失の初期化
    loss = Variable(ARR.zeros((), dtype='float32'))
    # <eos>をデコーダーに読み込ませる
    t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
    # デコーダーの計算
    for w in dec_words:
        # 1単語ずつをデコードする
        y, att, lambda_ = model.decode(t)
        # 正解単語をVariable型に変換
        t = Variable(ARR.array(w, dtype='int32'))

        # Generative Modeにより計算された単語のlog_softmaxをとる
        s = functions.log_softmax(y)
        # Attentionの重みのlog_softmaxをとる
        att_s = functions.log_softmax(att)
        # lambdaをsigmoid関数にかけることで、0~1の値に変更する
        lambda_s = functions.reshape(functions.sigmoid(lambda_), (batch_size,))
        # Generative Modeの損失の初期化
        Pg = Variable(ARR.zeros((), dtype='float32'))
        # Copy Modeの損失の初期化
        Pc = Variable(ARR.zeros((), dtype='float32'))
        # lambdaのバランスを学習するための損失の初期化
        epsilon = Variable(ARR.zeros((), dtype='float32'))
        # ここからバッチ内の一単語ずつの損失を計算する、for文を回してしまっているところがダサい・・・
        counter = 0
        for i, words in enumerate(w):
            # -1は学習しない単語につけているラベル
            if words != -1:
                # Generative Modeの損失の計算
                Pg += functions.get_item(functions.get_item(s, i), words) * functions.reshape((1.0 - functions.get_item(lambda_s, i)), ())
                counter += 1
                # もし入力文の中に出力したい単語が存在すれば
                if words in enc_key[i]:
                    # Copy Modeの計算をする
                    Pc += functions.get_item(functions.get_item(att_s, i), list(enc_key[i]).index(words)) * functions.reshape(functions.get_item(lambda_s, i), ())
                    # ラムダがCopy Modeよりになるように学習
                    epsilon += functions.log(functions.get_item(lambda_s, i))
                # 入力文の中に出力したい単語がなければ
                else:
                    # ラムダがGenerative Modeよりになるように学習
                    epsilon += functions.log(1.0 - functions.get_item(lambda_s, i))
        # それぞれの損失をバッチサイズで割って、合計する
        Pg *= (-1.0 / np.max([1, counter]))
        Pc *= (-1.0 / np.max([1, counter]))
        epsilon *= (-1.0 / np.max([1, counter]))
        loss += Pg + Pc + epsilon
    return loss

def forward_test(enc_words, model, ARR):
    ret = []
    ret_mode = []
    model.reset()
    enc_key = enc_words
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0], dtype='int32'))
    counter = 0
    while counter < 50:
        y, att, lambda_ = model.decode(t)
        lambda_ = functions.sigmoid(lambda_)
        s = functions.softmax(y)
        att_s = functions.softmax(att)
        prob = lambda_.data[0][0]
        flag = np.random.choice(2, 1, p=[1.0 - prob, prob])[0]
        #if prob > 0.5:
         #   flag = 1
        #else:
        #    flag = 0
        if  flag == 0:
            label = s.data.argmax()
            ret.append(label)
            ret_mode.append('gen')
            t = Variable(ARR.array([label], dtype='int32'))
        else:
            col = att_s.data.argmax()
            label = enc_key[col][0]
            ret.append(label)
            ret_mode.append('copy')
            t = Variable(ARR.array([label], dtype='int32'))
        counter += 1
        if label == 1:
            counter = 50
    return ret, ret_mode