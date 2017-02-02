# coding: utf-8
"""
Pachinko Allocation Modelの実装
Lin and MacCallum, 'Pachinko Allocation: DAG-Structured Mixture Models of Topic Correlations', ICML, 2006.
"""
import numpy as np
from scipy import special


class PAM(object):
    def __init__(self, bw, S=1, K=3, alpha0=1.0, alpha1=1.0, beta=1.0):
        """

        :param bw: bag-of-words
        :param S: 上位トピックの数
        :param K: 下位トピックの数
        :param alpha0: 上位トピックのハイパーパラメータの初期値
        :param alpha1: 下位トピックのハイパーパラメータの初期値
        :param beta: 単語分布のハイパーパラメータ
        """
        self.bw = bw
        self.S = S
        self.K = K
        # 上位トピックのハイパーパラメータ(1, S)
        self.alpha0 = np.zeros(self.S) + alpha0
        # 下位トピックのハイパーパラメータ(S, K)
        self.alpha1 = np.zeros(shape=(S, K)) + alpha1
        self.beta = beta
        # 語彙の数
        self.V = np.max([word for words in self.bw for word in words]) + 1
        # 文書数
        self.D = len(self.bw)
        # 各トピックの単語に割り当てられているトピック，(D, W, 2)ランダムに初期化
        self.z_s_k = [[[np.random.randint(0, self.S), np.random.randint(0, self.K)]
                       for w in words]
                      for words in self.bw]
        # ドキュメントごとに，Sトピック，Kトピックに割り当てられている単語の集計
        self.D_S_K = np.zeros(shape=(self.D, S, K))
        # 下位トピックごとにある単語の数
        self.K_V = np.zeros(shape=(self.K, self.V))
        # 単語分布
        self.phi = None
        # 上位トピック分布
        self.theta0 = None

    # D_S_Kの集計
    def set_params(self):
        for d, words in enumerate(self.bw):
            for w, word in enumerate(words):
                self.D_S_K[d][self.z_s_k[d][w][0]][self.z_s_k[d][w][1]] += 1
                self.K_V[self.z_s_k[d][w][1]][word] += 1

    # zのサンプリング
    def infer_z(self):
        for d, words in enumerate(self.bw):
            for w, word in enumerate(words):
                # 該当単語を抜く
                self.D_S_K[d][self.z_s_k[d][w][0]][self.z_s_k[d][w][1]] -= 1
                self.K_V[self.z_s_k[d][w][1]][word] -= 1
                # p(z = s, k)を計算する
                probs = np.zeros(self.S*self.K)
                for s in range(self.S):
                    for k in range(self.K):
                        N_ds = np.sum(self.D_S_K[d][s])
                        N_dsk = self.D_S_K[d][s][k]
                        N_k = np.sum(self.K_V[k])
                        N_kw = self.K_V[k][word]
                        #print N_dsk + self.alpha1[s][k]
                        #print N_ds + np.sum(self.alpha1[s])
                        prob = (N_ds + self.alpha0[s]) \
                             * ((N_dsk + self.alpha1[s][k]) / (N_ds + np.sum(self.alpha1[s]))) \
                             * ((N_kw + self.beta) / (N_k + self.beta * self.V))
                        probs[self.K*s + k] = prob
                probs /= np.sum(probs)
                # サンプリング
                s_k = np.random.multinomial(1, probs).argmax()
                # カウントの更新
                self.z_s_k[d][w][0] = s_k // self.K
                self.z_s_k[d][w][1] = s_k % self.K
                self.D_S_K[d][self.z_s_k[d][w][0]][self.z_s_k[d][w][1]] += 1
                self.K_V[self.z_s_k[d][w][1]][word] += 1

    # alpha0の推定(MLP version)
    def infer_alpha0(self):
        # alpha0の更新
        alpha0_base = np.zeros(self.S)
        for s in range(self.S):
            mole = np.sum([special.digamma(np.sum(self.D_S_K[d][s])+self.alpha0[s])
                           for d in range(self.D)]) \
                 - self.D * special.digamma(self.alpha0[s])
            deno = np.sum([special.digamma(np.sum(self.D_S_K[d])+np.sum(self.alpha0))
                           for d in range(self.D)]) \
                 - self.D * special.digamma(np.sum(self.alpha0))
            if mole != 0:
                alpha0_base[s] = mole / deno
            else:
                alpha0_base[s] = 1.0
        self.alpha0 *= alpha0_base

    # alpha1の推定(MLP version)
    def infer_alpha1(self):
        # alpha1の更新
        alpha1_base = np.zeros(shape=(self.S, self.K))
        for s in range(self.S):
            for k in range(self.K):
                mole = np.sum([special.digamma(self.D_S_K[d][s][k]+self.alpha1[s][k])
                               for d in range(self.D)]) \
                     - self.D * special.digamma(self.alpha1[s][k])
                deno = np.sum([special.digamma(np.sum(self.D_S_K[d][s])+np.sum(self.alpha1[s]))
                               for d in range(self.D)]) \
                     - self.D * special.digamma(np.sum(self.alpha1[s]))
                # alphaが0になることを防ぐため，もし分子が1になったら無視する
                if mole != 0:
                    alpha1_base[s][k] = mole / deno
                else:
                    alpha1_base[s][k] = 1.0
        self.alpha1 *= alpha1_base

    # betaの推定
    def infer_beta(self):
        mole = np.sum(special.digamma(self.K_V+self.beta)) \
             - self.K * self.V * special.digamma(self.beta)
        deno = self.V * np.sum(special.digamma(np.sum(self.K_V, axis=1)+self.beta*self.V)) \
             - self.K * self.V * special.digamma(self.beta*self.V)
        self.beta *= mole / deno


    # 計算部分
    def inference(self):
        # 単語のトピックのサンプリング
        self.infer_z()
        # ハイパーパラメータalphaの更新
        self.infer_alpha0()
        self.infer_alpha1()
        # ハイパーパラメータbetaの更新
        self.infer_beta()

    # 単語分布の計算
    def cal_phi(self):
        self.phi = (self.K_V + self.beta) / (np.sum(self.K_V, axis=1)[:,np.newaxis] + self.beta * self.V)

    # 単語分布の取得
    def get_phi(self):
        if self.phi == None:
            self.cal_phi()
        return self.phi

    # 上位トピック分布の計算
    def cal_theta0(self):
        mole = np.array([np.sum(S_K, axis=1)+self.alpha0 for S_K in self.D_S_K])
        deno = np.sum(mole, axis=1)[:, np.newaxis]
        self.theta0 = mole / deno

    # 上位トピックの取得
    def get_theta0(self):
        if self.theta0 == None:
            self.cal_theta0()
        return self.theta0

    # ハイパーパラメータalpha1の取得
    def get_alpha1(self):
        return self.alpha1

# メイン関数
def main():
    # 9つの文章に対して、９つの単語、３つのトピック
    bag_of_word = np.array([[0, 1, 2],
                            [0, 1, 2],
                            [0, 1, 2, 2],
                            [3, 5, 5, 1],
                            [3, 3, 4, 4, 0],
                            [3, 4, 4, 5],
                            [6, 7, 8],
                            [6, 7, 7, 8],
                            [7, 7, 8, 8]])
    # 上位トピックの数
    S = 2
    # 下位トピックの数
    K = 3
    # 上位トピックのパラメータの初期値
    alpha0 = 0.1
    # 下位トピックのパラメータの初期値
    alpha1 = 0.1
    # 単語分布の初期値
    beta = 0.1
    # イテレーションの回数
    num_iter = 300

    # インスタンス化
    pam = PAM(bw=bag_of_word,
              S=S,
              K=K,
              alpha0=alpha0,
              alpha1=alpha1,
              beta=beta)

    # パラメータの初期化
    pam.set_params()

    # 推定
    for iter_ in range(num_iter):
        pam.inference()

    # 単語分布の表示
    print "====単語分布===="
    print pam.get_phi()
    # 上位トピック分布の表示
    print "====上位トピック分布===="
    print pam.get_theta0()
    # トピックの関連性の表示
    print "トピックの関連性"
    print pam.get_alpha1()

# 実行
if __name__ == '__main__':
    main()