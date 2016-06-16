# coding: utf-8
"""
GSDMMのサンプル
出展：
'A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering'
Jianhua Yin, Jianyong Wang, ACM, SIGKDD, 2014

使い方：
gsdmm = GSDMM(alpha=0.1, beta=0.1, K=100, I=20)
gsdmm.set_param(path='path/to/file' or list_d_words=list_list_word)
gsdmm.fit()
list_prob = gsdmm.infer(list_words)

学習用のファイル:
sample.txt
words in a document in one line sep by single space
"""

import numpy as np


class GSDMM():
    def __init__(self, alpha=0.1, beta=0.1, K=100, I=20):
        # 必要なパラメータ
        self.alpha = alpha
        self.beta = beta
        # number of initial K
        self.K = K
        # number of iterations
        self.I = I

        # 初期パラメータ以外を設定
        self.list_bags = None
        # number of vocaburary
        self.V = None
        # number of documents
        self.D = None
        # cluster labels of each documents
        self.list_z = None
        # number of documents in cluster z
        self.list_mz = None
        # number of words in cluster z
        self.list_nz = None
        # number of occurences of word w in cluster z
        self.list_z_w = None

        # dictの作成
        self.dict_word_id = None
        self.dict_id_word = None
        # number of occurrence of word w in document d
        self.list_d_w = None
        # number of words in document d
        self.list_Nd = None

        # 分布
        self.list_theta = None
        self.list_dict_phi = None

    # parameterの設定
    def set_param(self, path=None, list_d_words=None):
        # pathセットされたら読み込み、listがセットされたら代入
        if path != None:
            self.list_bags = self.readtxt(path)
        elif list_d_words != None:
            self.list_bags = list_d_words
        else:
            print 'Error: pathかlistをセットしてください'
            return None

        # number of vocaburary
        self.V = len(set([word for row in self.list_bags for word in row]))
        # number of documents
        self.D = len(self.list_bags)
        # cluster labels of each documents
        self.list_z = np.array([0 for i in range(self.D)], dtype=int)
        # number of documents in cluster z
        self.list_mz = np.array([0 for i in range(self.K)], dtype=int)
        # number of words in cluster z
        self.list_nz = np.array([0 for i in range(self.K)], dtype=int)
        # number of occurences of word w in cluster z
        self.list_z_w = np.array([[0 for j in range(self.V)]
                                  for i in range(self.K)], dtype=int)

        # dictの作成
        self.dict_word_id, self.dict_id_word = self.make_dict(self.list_bags)
        # number of occurrence of word w in document d
        self.list_d_w = self.cal_d_w(self.list_bags)
        # number of words in document d
        self.list_Nd = np.array([np.sum(row) for row in self.list_d_w], dtype=int)

    # word_id, id_wordのdict作成
    def make_dict(self, list_bags):
        list_words = [word for row in list_bags for word in row]
        dict_word_id = {word: i for i, word in enumerate(set(list_words))}
        dict_id_word = {i: word for i, word in enumerate(set(list_words))}
        return dict_word_id, dict_id_word

    # テキストファイルの読み込み
    def readtxt(self, path):
        f = open(path)
        lines = f.readlines()
        f.close
        list_bags = [row.rstrip('\r\n').split(" ") for row in lines]
        return list_bags

    # 初期化
    def initialize(self, d, z):
        self.list_z[d] = z
        self.list_mz[z] += 1
        self.list_nz[z] += self.list_Nd[d]
        non_zero_index = np.nonzero(self.list_d_w[d])[0]
        for j in non_zero_index:
            self.list_z_w[z][j] += self.list_d_w[d][j]

    # 式4の肝心な部分、dはドキュメント番号、zはクラスタの番号
    def cal_pz(self, d, z):
        # 式4の左側、オーバーフロー対策でlogで記述
        first = np.log((self.list_mz[z] + self.alpha))
        first -= np.log(np.float(self.D - 1 + self.K*self.alpha))
        # 式5の右側を分子と分母別々に計算、logで記述
        second = 0
        non_zero_index = np.nonzero(self.list_d_w[d])[0]
        for i in non_zero_index:
            for j in range(self.list_d_w[d][i]):
                second += np.log((self.list_z_w[z][i] + self.beta + j))
        for i in range(self.list_Nd[d]):
            second -= np.log(self.list_nz[z] + self.V*self.beta + i)
        # 右側と左側を足したものを返す（log）
        return first + second

    # ギブスサンプリングのための変数を抜く
    def uninitialize(self, d, z):
        self.list_mz[z] -= 1
        self.list_nz[z] -= self.list_Nd[d]
        non_zero_index = np.nonzero(self.list_d_w[d])[0]
        for j in non_zero_index:
            self.list_z_w[z][j] -= self.list_d_w[d][j]

    # トピック分布の推定
    def cal_theta(self):
        list_theta = [float(self.list_mz[i]+self.alpha)/(self.D+self.K*self.alpha)
                      for i in range(self.K)]
        return list_theta

    # 単語分布の推定
    def cal_phi(self):
        list_phi = [[float(self.list_z_w[i][j]+self.beta)/(self.list_nz[i] + self.V*self.beta)
                     for j in range(self.V)] for i in range(self.K)]
        list_dict_phi = [{self.dict_id_word[j]: phi
                          for j, phi in enumerate(list_phi[i])}
                         for i in range(self.K)]
        return list_dict_phi

    # list_d_wの作成
    def cal_d_w(self, list_bags):
        list_d_w = np.array([[0 for i in range(self.V)] for i in range(self.D)], dtype=int)
        for i, row in enumerate(list_bags):
            for word in row:
                list_d_w[i][self.dict_word_id[word]] += 1
        return list_d_w

    # コーパスを元に学習
    def fit(self):
        for d in range(self.D):
            z = np.random.choice(self.K)
            self.initialize(d, z)

        for iteration in range(self.I):
            for d in range(self.D):
                self.uninitialize(d, self.list_z[d])
                log_prob = np.array([self.cal_pz(d, z) for z in range(self.K)])
                max_log = np.amax(log_prob)
                log_prob -= max_log
                # logを戻す
                prob = np.exp(log_prob)
                prob /= np.sum(prob)
                # 新しいラベルをサンプリング
                z = np.random.choice(self.K, p=prob)
                self.initialize(d, z)
            print "finish: ", iteration, " iteration"

        self.list_theta = self.cal_theta()
        self.list_dict_phi = self.cal_phi()

        print "all finished"

    # 学習した結果を元にコーパスのクラスを推定
    def infer(self, list_words):
        try:
            # 学習が先に行われていなければエラーを上げる
            if self.list_theta == None:
                raise NameError('Please fit first')
            # すべての単語が辞書に含まれていなければエラーを上げる
            for word in list_words:
                if word in self.dict_word_id:
                    break
            else:
                raise KeyError('No word found in dict')

            list_overflow = []
            for i in range(self.K):
                prob = 0.0
                prob += np.log(self.list_theta[i])
                for word in list_words:
                    if word in self.dict_word_id:
                        prob += np.log(self.list_dict_phi[i][word])
                list_overflow.append(prob)
            log_max = np.max(list_overflow)
            list_overflow = np.array(list_overflow) - log_max
            # 正規化
            list_prob = np.exp(list_overflow)/np.sum(np.exp(list_overflow))
            return list_prob

        except NameError:
            raise
        except KeyError:
            raise
