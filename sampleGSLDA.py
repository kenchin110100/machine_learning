# coding: utf-8
"""
LDAコーディング（ギブスサンプリング）
"""

import numpy as np
import copy
from collections import Counter

bag_of_word = np.array([[0, 1, 2],
                        [0, 1, 2],
                        [0, 1, 2, 2],
                        [3, 4, 5],
                        [3, 3, 4, 5],
                        [3, 4, 4, 5],
                        [6, 7, 8],
                        [6, 7, 7, 8],
                        [7, 7, 8, 8]])


class LDA(object):
    def __init__(self, bw, K=3, alpha=1.0, beta=1.0):
        self.bw = bw
        self.K = K
        self.alpha = alpha
        self.beta = beta
        # 語彙数
        self.V = np.argmax(self.bw) + 1
        # 文書数
        self.D = len(self.bw)
        # 文書ごとのトピック分布
        self.k_d = None
        # トピックごとの単語分布
        self.v_k = None
        # 単語に割り当てられたトピック
        self.w_z = np.array([[0 for i in row] for row in self.bw])
        # トピック分布，単語分布の初期化
        self.initialize()

    def initialize(self):
        # 文書ごとのトピック分布を初期化
        self.k_d = np.random.rand(self.D, self.K)
        self.k_d /= np.sum(self.k_d, axis=1)[:,np.newaxis]
        # トピックごとの単語分布を初期化
        self.v_k = np.random.rand(self.K, self.V)
        self.v_k /= np.sum(self.v_k, axis=1)[:, np.newaxis]

    def inference(self):
        # トピック分布，単語分布を用いて各単語のトピックをサンプリング
        for d, doc in enumerate(self.bw):
            for w, word in enumerate(doc):
                dist = self.v_k[:,word] * self.k_d[d]
                dist /= np.sum(dist)
                self.w_z[d][w] = np.random.choice(self.K, 1, p=dist)[0]

        # 各単語のトピックからトピック分布をサンプリング
        for d, topics in enumerate(self.w_z):
            topic_num = Counter(topics)
            n_d = [topic_num[k]+self.alpha
                   if k in topic_num else self.alpha
                   for k in range(self.K)]
            self.k_d[d] = np.random.dirichlet(n_d)

        # 各単語のトピックから単語分布の推定
        words_topic = np.zeros((self.K, self.V))
        for doc, topics in zip(self.bw, self.w_z):
            for word, topic in zip(doc, topics):
                words_topic[topic][word] += 1
        for k in range(self.K):
            self.v_k[k] = np.random.dirichlet(words_topic[k]+self.beta)


    def get_topics(self):
        return self.k_d

    def get_words(self):
        return self.v_k


def main():
    lda = LDA(bag_of_word, K=3, alpha=0.1, beta=0.1)
    for i in range(300):
        lda.inference()
    print "====トピック分布===="
    print lda.get_topics()
    print "===================="
    print "======単語分布======"
    print lda.get_words()


if __name__ == '__main__':
    main()
