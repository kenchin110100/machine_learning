# coding: utf-8
"""
LDAの実装（PYMCを使って）
未完成
"""

import numpy as np
import pymc as pm

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


class LDA(object):
    def __init__(self, bw, K=3, alpha=0.1, beta=0.1):
        self.bw = bw
        # トピック数
        self.K = K
        # 文書数
        self.D = len(self.bw)
        # 語彙数
        self.V = np.argmax(self.bw)+1
        # ハイパーパラメータalpha
        self.alpha = np.ones(self.K) * alpha
        # ハイパーパラメータbeta
        self.beta = np.ones(self.V) * beta
        # 結果が格納されるインスタンス
        self.mcmc = None

    def inference(self, iter_=100):
        theta = pm.Container([pm.CompletedDirichlet("theta_%s"%d,
                                                    pm.Dirichlet("ptheta_%s"%d,
                                                                 theta=self.alpha)
                                                    )
                              for d in range(self.D)])
        phi = pm.Container([pm.CompletedDirichlet("phi_%s"%k,
                                                  pm.Dirichlet("pphi_%s"%k,
                                                               theta=self.beta))
                            for k in range(self.K)])
        z_d = pm.Container([pm.Categorical("z_%s"%d,
                                           p=theta[d],
                                           size=len(self.bw[d]))
                            for d in range(self.D)])
        w_z = pm.Container([pm.Categorical("w_%s_%s"%(d, w),
                                           p=phi[z_d[d][w].get_value()],
                                           value=self.bw[d][w],
                                           observed=True)
                            for d in range(self.D) for w in range(len(self.bw[d]))])

        model = pm.Model([theta, phi, z_d, w_z])
        self.mcmc = pm.MCMC(model)
        self.mcmc.sample(iter_)

    def get_words(self):
        return np.array([list(self.mcmc.trace("phi_%s"%k)[-1][0]) for k in range(self.K)], dtype='float16')

    def get_topics(self):
        return np.array([list(self.mcmc.trace("theta_%s"%d)[-1][0]) for d in range(self.D)], dtype='float16')


def main():
    lda = LDA(bag_of_word, K=3, alpha=0.1, beta=0.1)
    lda.inference(iter_=300)
    print "====トピック分布===="
    print lda.get_topics()
    print "===================="
    print "======語彙分布======"
    print lda.get_words()


if __name__ == '__main__':
    main()
