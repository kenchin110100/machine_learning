# coding: utf-8
"""
Attention Model + LSTMを使って文書分類をするコード
"""

import numpy as np
from chainer import Chain, Variable, cuda, functions, links

from sampleSeq2Sep import LSTM_Encoder


class LstmEncoder(Chain):
    def __init__(self, input_size, hidden_size):
        """
        クラスの初期化
        :param imput_size: 入力されるデータの次元数
        :param hidden_size: 隠れ層のサイズ
        """
        super(LstmEncoder, self).__init__(
            eh = links.Linear(input_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        """

        :param x: 入力ベクトル
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        e = functions.tanh(x)
        return functions.lstm(c, self.eh(e) + self.hh(h))


class Attention(Chain):
    def __init__(self, hidden_size, flag_gpu, flag_train):
        """
        Attentionのインスタンス化
        :param hidden_size: 隠れ層のサイズ(int)
        :param flag_gpu: GPUを使うかどうか(bool)
        :param flag_train: 学習かテストか(bool)
        """
        super(Attention, self).__init__(
            # 入力されたベクトルの線形結合層
            fh=links.Linear(hidden_size, hidden_size),
            # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
            hw=links.Linear(hidden_size, 1),
        )
        # 隠れ層のサイズを記憶
        self.hidden_size = hidden_size
        # trainかtestかのflag
        self.flag_train = flag_train
        # GPUを使う場合はcupyを使わないときはnumpyを使う
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

    def __call__(self, inputs, masks):
        """
        Attentionの計算
        :param inputs: LSTMでエンコードされたベクトルのリスト
        :param masks: maskする列が0、しない列が1と記録されたリスト
        :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
        """
        # ミニバッチのサイズを記憶
        batch_size = h.data.shape[0]
        # ウェイトを記録するためのリストの初期化
        ws = []
        # ウェイトの合計値を計算するための値を初期化
        sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for i, mask in inputs, masks:
            # 入力されたベクトルを線型結合して、中間ベクトルを作成
            w = functions.tanh(self.fh(i))
            # 中間ベクトルからウェイトを計算してsoftmaxを計算するためにexpする
            w = functions.exp(self.hw(w))
            # 計算したウェイトを記録、ここでmaskも同時に行う
            ws.append(w * mask)
            sum_w += w * mask
        # 出力する加重平均ベクトルの初期化
        att_vecs = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        for num, (i, w) in enumerate(zip(inputs, ws)):
            # ウェイトの和が1になるように正規化
            w /= sum_w
            ws[num] = w
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_vecs += functions.reshape(functions.batch_matmul(i, w), (batch_size, self.hidden_size))
        # trainならベクトルのみを、テストならベクトルと重みを返す
        if self.flag_train == True:
            return att_vecs
        else:
            return att_vecs, ws


class AttClassifier(Chain):
    def __init__(self, input_size, hidden_size, batch_size, label_size, flag_gpu=True, flag_train=True):
        """
        LSTM + Attentionのインスタンス化
        :param input_size: 入力ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(AttClassifier, self).__init__(
            # 順向きのEncoder
            f_encoder = LstmEncoder(input_size, hidden_size),
            # 逆向きのEncoder
            b_encoder = LstmEncoder(vocab_size, hidden_size),
            # Attention Model
            attention = Attention(hidden_size*2, flag_gpu, flag_train),
            # 加重平均されたベクトルからラベルサイズのベクトルを出力
            predictor = links.Linear(hidden_size*2, label_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.flag_train = flag_train

        # GPUを使うときはcupy、使わないときはnumpy
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fs = []
        self.bs = []
        self.hs = []

    def encode(self, inputs):
        """
        Encoderの計算
        :param inputs: 入力されるベクトル列が記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # 先ずは順向きのEncoderの計算
        for i in inputs:
            c, h = self.f_encoder(i, c, h)
            # 計算された中間ベクトルを記録
            self.fs.append(h)

        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # 逆向きのEncoderの計算
        for i in reversed(inputs):
            c, h = self.b_encoder(i, c, h)
            # 計算された中間ベクトルを記録
            self.bs.insert(0, h)

        # 順向きのEncoderと逆向きのEncoderのベクトルをコンカットする
        for f, b in zip(self.fs, self.bs):
            self.hs.append(functions.concat([f, b]))


    def predict(self, masks):
        """
        Decoderの計算
        :param masks: 入力されたベクトルのどの部分をマスクするか記録したベクトル
        :return: 2次元のベクトル
        """
        # trainの場合は加重平均ベクトルのみを、testの場合はウェイトも返す
        if self.flag_train == True:
            att_vecs, _ = self.attention(self.hs, masks)
            return self.predictor(functions.tanh(att_vecs))
        else:
            att_vecs, weights = self.attention(self.hs, masks)
            return self.predictor(functions.tanh(att_vecs)), weights


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
        self.hs = []
        # 勾配の初期化
        self.zerograds()


class DatasetIterator(Object):
    def __init__(self, x_train, y_train, vec_size, batch_size):
        """
        データセットクラス
        イテレータを回すとデータが出てくるようにする
        :param x_train: 説明変数
        :param y_train: 非説明編数
        :param vec_size: 説明変数のサイズ
        :param batch_size: ミニバッチのサイズ
        """
        self.vec_size = vec_size
        self.batch_size = batch_size
        self.num_data = len(x_train)
        self.count = 0

        # 初期化
        self.xy_s = self.set_params()
        # シャッフル
        self.shuffle()


    def __iter__(self):
        return self


    def next(self):
        if self.count * batch_size > self.num_data:
            raise StopIteration()
        # バッチサイズの文のデータを取り出す
        xy_s = self.xy_s[self.count * self.batch_size: (self.count+1) * self.batch_size]
        # xのデータ、yのデータだけ取り出す
        xs = [xy['x'] for xy in xy_s]
        ys = [xy['y'] for xy in xy_s]
        # LSTMで並列処理をするために、xsの系列長を揃える
        xs, masks = self.get_reshaped_xs(xs)
        # xs, ys, masksをnumpy.array型に変換する。
        # またxs, masksに関しては転置する
        xs = np.transpose(np.array(xs, dtype='float32'), (1,0,2))
        masks = np.transpose(np.array(masks, dtype='float32'))
        ys = np.array(ys, dtype='int32')

        return (xs, ys, masks)


    def get_reshaped_xs(self, xs):
        """
        ミニバッチ内の系列長の長さを揃える
        """
        # xsの中で一番要素数が多いものを取得
        max_len = self.get_max_len(xs)
        # 中埋めする値
        padding = np.zeros(self.vec_size)
        masks = [[1 if l < len(x) else 0 for l in range(max_len)]
                 for x in xs]
        reshaped_xs = [[x[l] if l < len(x) else padding for l in range(max_len)]
                       for x in xs]
        return reshaped_xs, masks


    def get_max_len(self, lists):
        """
        list_list型で、要素数が一番大きい値を返す
        """
        return np.max([len(l)for l in lists])


    def set_params(self, x_train, y_train):
        """
        データセットの前処理
        """
        assert len(x_train) != len(y_train), "XとYの長さが違います"

        # シャッフルできるようにXとYのデータを繋げる
        xy_s = [{'x': x, 'y': y} for x, y in zip(x_train, y_train)]

        return xy_s


    def shuffle():
        random.shuffle(self.xy_s)
        self.count = 0


def forward(inputs, masks, labels, model, ARR):
    """
    順伝播の計算
    :params inputs: 入力ベクトル
    :params masks: マスクベクトル
    :params labels: 正解ラベル
    :params model: インスタンス化したモデル
    :params ARR: cupyかnumpy
    """
    # バッチサイズの確認
    batch_size = len(enc_words[0])
    # モデルのリセット
    model.reset()
    # 入力ベクトルのVariable化
    inputs = [Variable(ARR.array(row, dtype='float32')) for row in inputs]
    # maskベクトルのVariable化
    masks = [Variable(ARR.array(row, dtype='float32')) for row in masks]
    # 正解ラベルのVariable化
    labels = Variable(ARR.array(labels, dtype='int32'))

    # LSTMにより、中間層ベクトルの計算
    model.encode(inputs)
    # 中間層ベクトルの加重平均からラベルを予測
    ys = model.predict(masks)
    # 損失の計算
    loss = functions.softmax_cross_entropy(ys, labels)
    return loss


def forward_test(inputs, masks, model, ARR):
    """
    順伝播の計算（予測用）
    :params inputs: 入力ベクトル
    :params masks: マスクベクトル
    :params model: インスタンス化したモデル
    :params ARR: cupyかnumpy
    """
    # バッチサイズの確認
    batch_size = len(enc_words[0])
    # モデルのリセット
    model.reset()
    # 入力ベクトルのVariable化
    inputs = [Variable(ARR.array(row, dtype='float32')) for row in inputs]
    # maskベクトルのVariable化
    masks = [Variable(ARR.array(row, dtype='float32')) for row in masks]
    # modelのリセット
    model.reset()
    # 入力ベクトルのエンコード
    model.encode(inputs)
    # 予測ラベルとウェイトの出力
    ys, weights = model.predict(masks)

    return functions.softmax(ys), weights


def train():

    # モデルのインスタンス化
    model = AttClassifier(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            label_size=LABEL_SIZE,
            flag_gpu=FLAG_GPU,
            flag_train=True
            )

    # モデルの初期化
    model.reset()

    # データセットクラスの初期化
    corpuses = Dataset(x_train, y_train, BATCH_SIZE)

    # エポックの分だけ学習を回す
    for epoch in range(NUM_EPOCH):
        # Adamの初期化
        opt = optimizers.Adam()
        # optimizerの初期化
        opt.setup(model)
        # grad_clippingの設定
        opt.add_hook(optimizer.GradientClipping(CLIP_SIZE))

        # データセットのシャッフル
        corpuses.shuffle()

        # バッチ学習開始
        for x_corpus, y_corpus, masks in corpuses:
            # 順伝播の計算
            loss = forward(
                    inputs=x_corpus,
                    masks=masks,
                    labels=y_corpus,
                    model=model,
                    ARR=np)
            # バックプロパゲーション
            loss.backward()
            # 勾配の更新
            opt.update()
            # オプティマイザーの初期化
            opt.zero_grads()

        # エポックごとにモデルを保存
        serializers.save_npz(OUTPUT_PATH%epoch, model)

