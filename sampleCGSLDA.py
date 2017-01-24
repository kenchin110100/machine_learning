# coding: utf-8
"""
LDAコーディングの練習(周辺化ギブスサンプリング)
"""
import numpy as np
import copy

# 9つの文章に対して、９つの単語、３つのトピック
bag_of_word = np.array([[0, 1, 2],
                        [0, 1, 2],
                        [0, 1, 2, 2],
                        [3, 4, 5],
                        [3, 3, 4, 5],
                        [3, 4, 4, 5],
                        [6, 7, 8],
                        [6, 7, 7, 8],
                        [7, 7, 8, 8]])


class LDA:
    def __init__(self, bw, k=3, alpha=1.0, beta=1.0):
        self.bw = bw        # bag of words
        self.k = k          # num of topic
        self.alpha = alpha  # alpha
        self.beta = beta    # beta
        self.v = np.argmax(self.bw) + 1     # size of vocaburary
        # p(k|d)　ドキュメントがトピックkの確率
        self.k_d = np.zeros((self.bw.shape[0], self.k)) + self.alpha
        # p(w|k)  トピックkからwが発生する確率
        self.w_k = np.zeros((self.k, self.v)) + self.beta
        # 各文書内の各単語に、ランダムでトピックを決める
        self.word_topic = np.random.randint(0, self.k, (self.bw.shape[0], self.v))
        # トピックの割合
        self.z = np.zeros((1, self.k)) + self.v * self.beta

    def set_corpus(self):
        # 文章をトピック（の多項分布）で表現する
        for d, doc in enumerate(self.bw):
            for v in doc:
                self.k_d[d][self.word_topic[d][v]] += 1
                self.w_k[self.word_topic[d][v]][v] += 1
                self.z[0][self.word_topic[d][v]] += 1

    def inference(self):
        for d, doc in enumerate(self.bw):
            for v in doc:
                old_z = self.word_topic[d][v]
                self.k_d[d][old_z] -= 1
                self.w_k[old_z][v] -= 1
                self.z[0][old_z] -= 1
                p_k_wd = self.k_d[d] * self.w_k[:,v] / self.z[0]
                new_z = np.random.multinomial(100, p_k_wd / np.sum(p_k_wd)).argmax()
                self.word_topic[d][v] = new_z
                self.k_d[d][new_z] += 1
                self.w_k[new_z][v] += 1
                self.z[0][new_z] += 1

    def wordtopic(self):
        return self.w_k / self.z.T

    def topicdocument(self):
        return self.k_d / np.sum(self.k_d, axis=1)[:, np.newaxis]


lda = LDA(bag_of_word, k=3)

lda.set_corpus()
for i in range(100):
    lda.inference()

print lda.wordtopic()
