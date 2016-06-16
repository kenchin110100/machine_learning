# coding: utf-8
"""
Unigram Mixtureのサンプル
出展：
'Text Classfication from Labeled and Unlabeled Documents using EM'
K. Nigam, A. K. McCallum and S. Mitchell, Machine Learning, 2000

使い方：
um = UM(alpha=1.0, beta=1.0, K=10, converge=0.01, max_iter=100)
um.set_param(path='path/to/file' or list_d_words=list_list_words)
um.fit()
list_prob = um.infer(list_words)

学習用のファイル:
sample.txt
words in a document in one line sep by single space
"""

import numpy as np


class UM():
    def __init__(self, alpha=1.0, beta=1.0, K=10, converge=0.01, max_iter=100):
        # 必要なパラメータ
        self.alpha = alpha
        self.beta = beta
        # number of initial K
        self.K = K
        # number of iterations
        self.converge = converge
        # max_iteration
        self.max_iter = max_iter

        self.list_bags = None
        # number of vocaburary
        self.V = None
        # number of documents
        self.D = None
        # likelihood
        self.likelihood = None

        # dictの作成
        self.dict_word_id = None
        self.dict_id_word = None
        # number of occurrence of word w in document d
        self.list_d_w = None
        # 負担率
        self.list_qdk = None

        # 分布
        self.list_theta = None
        self.list_phi = None
        self.list_dict_phi = None

    # parameterの作成
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

        # dictの作成
        self.dict_word_id, self.dict_id_word = self.make_dict(self.list_bags)
        # number of occurrence of word w in document d
        self.list_d_w = self.cal_d_w(self.list_bags)
        # number of words in document d
        self.list_Nd = np.array([np.sum(row) for row in self.list_d_w], dtype=int)
        # 負担率
        self.list_qdk = np.array([[0 for j in range(self.K)] for i in range(self.D)], dtype=float)

    # word_id, id_wordのdict作成
    def make_dict(self, list_bags):
        list_words = [word for row in list_bags for word in row]
        dict_word_id = {word: i for i, word in enumerate(set(list_words))}
        dict_id_word = {i: word for i, word in enumerate(set(list_words))}
        return dict_word_id, dict_id_word

    # テキストファイルの読み込み
    def readtxt(self, path, LF='\n'):
        f = open(path)
        lines = f.readlines()
        f.close
        list_bags = [row.rstrip(LF).split(" ") for row in lines]
        return list_bags

    # 初期化
    def initialize(self):
        self.list_theta = np.random.dirichlet([self.alpha] * self.K)
        self.list_phi = np.random.dirichlet([self.alpha] * self.V, self.K)

    # Estep: 負担率の計算
    def e_step(self):
        for d in range(self.D):
            # オーバーフロー対策
            list_overflow = np.array([0.0 for i in range(self.K)], dtype=float)
            for z in range(self.K):
                qdk = np.log(self.list_theta[z])
                non_zero_index = np.nonzero(self.list_d_w[d])[0]
                for i in non_zero_index:
                    qdk += self.list_d_w[d][i] * np.log(self.list_phi[z][i])
                list_overflow[z] = qdk
            max_log = np.max(list_overflow)
            list_overflow -= max_log
            sum_qdk_d = np.sum([np.exp(num) for num in list_overflow])
            list_overflow -= np.log(sum_qdk_d)
            self.list_qdk[d] = np.array([np.exp(num) for num in list_overflow])

    # Mstep: thetaとphiの更新
    def m_step(self):
        self.list_theta = np.array([(self.alpha + np.sum(self.list_qdk[:,z]))/(self.alpha*self.K + self.D)
                                   for z in range(self.K)])
        self.list_phi = np.array([[self.beta + np.sum(self.list_qdk[:,z] * self.list_d_w[:,w])
                                   for w in range(self.V)]
                                  for z in range(self.K)])
        self.list_phi /= np.reshape(np.sum(self.list_phi, axis=1), (self.K, 1))

    # stop判定するためのconvergeの計算
    def cal_likelihood(self):
        list_likelihood = []
        for d in range(self.D):
            non_zero_index = np.nonzero(self.list_d_w[d])[0]
            # オーバーフロー対策
            list_overflow = []
            l_document = 0.0
            for z in range(self.K):
                # print self.list_theta
                l_document += np.log(self.list_theta[z])
                for i in non_zero_index:
                    l_document += self.list_d_w[d][i] * np.log(self.list_phi[z][i])
                list_overflow.append(l_document)
            max_log = np.max(list_overflow)
            likelihood = 0.0
            for l_document in list_overflow:
                likelihood += np.exp(l_document - max_log)
            likelihood = np.log(likelihood) + max_log
            list_likelihood.append(likelihood)

        return np.sum(list_likelihood)/len(list_likelihood)


    # list_d_wの作成
    def cal_d_w(self, list_bags):
        list_d_w = np.array([[0 for i in range(self.V)] for i in range(self.D)], dtype=int)
        for i, row in enumerate(list_bags):
            for word in row:
                list_d_w[i][self.dict_word_id[word]] += 1
        return list_d_w

    # メインの学習部分
    def fit(self):
        self.initialize()
        self.likelihood = 0
        likelihood_tmp = self.cal_likelihood()
        for i in range(self.max_iter):
            self.likelihood = likelihood_tmp
            self.e_step()
            self.m_step()
            likelihood_tmp = self.cal_likelihood()
            if i % 10 == 0:
                print 'finish: ', i+1, ' iteration'
                print 'likelihood: ', likelihood_tmp
            if np.fabs(self.likelihood - likelihood_tmp) < self.converge:
                break
        self.likelihood = likelihood_tmp
        self.list_dict_phi = [{self.dict_id_word[i]: phi
                               for i, phi in enumerate(self.list_phi[z])}
                              for z in range(self.K)]
        print 'finish all: ', self.likelihood

    # クラスの推定
    def infer(self, list_words):
        try:
            # 学習が先に行われていなければエラーを上げる
            if self.list_theta == None:
                raise NameError('calculation first')
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
                        prob += self.list_dict_phi[i][word]
                list_overflow.append(prob)
            max_log = np.max(list_overflow)
            list_overflow = [np.exp(num - max_log) for num in list_overflow]
            # 正規化
            list_prob = np.array(list_overflow)/np.sum(list_overflow)
            return list_prob

        except NameError:
            raise
        except KeyError:
            raise
