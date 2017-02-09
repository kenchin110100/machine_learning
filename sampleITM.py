# coding: utf-8
"""
Interractive Topic Modelの実装
Y.Hu, J.B.Graber, B.Satinoff, "Interractive Topic Model", ACL, 2011
"""
import numpy as np
import copy

class ITM:
    def __init__(self, bw, K=3, alpha=0.1, beta=0.01, eta=100.):
        # bag of words
        self.bw = bw
        # 制約を追加したbag of words
        self.bw_c = bw
        # トピック数
        self.K = K
        # 語彙数
        self.V = np.max([word for words in self.bw for word in words]) + 1
        # 文書数
        self.D = len(self.bw)
        # トピック分布のハイパーパラメータ
        self.alpha = alpha
        # 単語分布のハイパーパラメータ
        self.beta = beta
        # 制約のハイパーパラメータ
        self.eta = eta
        # 制約をかける単語のリスト
        self.const = []
        # 制約数
        self.C = len(self.const)
        # 各文書の単語に割り当てられているトピック
        self.z_k = [[np.random.randint(0, self.K)
                     for word in words]
                    for words in self.bw]
        # トピックKに割り当てられた単語数(K, V)
        self.K_V = np.zeros(shape=(self.K, self.V))
        # トピックKに割り当てられた制約C内で語彙Vが出現した回数(K, C, V)
        self.K_C_V = []
        # 文書D内で、トピックKに割り当てられた単語の数(D, K)
        self.D_K = np.zeros(shape=(self.D, self.K))

    # K_V, D_Kの集計
    def set_params(self):
        for d, (words, zs) in enumerate(zip(self.bw, self.z_k)):
            for w, z in zip(words, zs):
                self.D_K[d][z] += 1
                self.K_V[z][w] += 1

    # 単語wが制約に含まれるかどうか調べる
    def search_const(self, w):
        for const_id, const_one in enumerate(self.const):
            if w in const_one:
                return True, const_id
                break
        else:
            return False, None


    def infer_z(self):
        for d, (words, zs) in enumerate(zip(self.bw, self.z_k)):
            for n, (w, z) in enumerate(zip(words, zs)):
                """
                d: 文書id
                n: 単語id
                w: 単語
                z: トピック
                """
                # 該当単語を抜く
                self.K_V[z][w] -= 1
                self.D_K[d][z] -= 1
                # 制約化の単語を抜く
                bool_, const_id = self.search_const(w)
                if bool_:
                    self.K_C_V[z][const_id][w] -= 1

                # p(z=k|Z,W)の計算
                probs = np.zeros(self.K)
                # 単語wが制約下にある場合
                if bool_:
                    for k in range(self.K):
                        T_dk = self.D_K[d][k]
                        T_d = np.sum(self.D_K[d])
                        P_kl = np.sum(self.K_C_V[k][const_id])
                        P_k = np.sum(self.K_V[k])
                        C_l = len(self.const[const_id])
                        W_klw = self.K_C_V[k][const_id][w]
                        W_kl = np.sum(self.K_C_V[k][const_id])
                        prob = (T_dk + self.alpha) / (T_d + self.alpha * self.K) \
                             * (P_kl + C_l*self.beta) / (P_k + self.V * self.beta) \
                             * (W_klw + self.eta) / (W_kl + C_l * self.eta)
                        probs[k] = prob
                # 単語wが制約下にない場合
                else:
                    for k in range(self.K):
                        T_dk = self.D_K[d][k]
                        T_d = np.sum(self.D_K[d])
                        P_kw = self.K_V[k][w]
                        P_k = np.sum(self.K_V[k])
                        prob = (T_dk + self.alpha) / (T_d + self.K * self.alpha) \
                             * (P_kw + self.beta) / (P_k + self.V * self.beta)
                        probs[k] = prob
                # 正規化してサンプリング
                probs /= np.sum(probs)
                # サンプリング
                z_new = np.random.multinomial(1, probs).argmax()
                # 新しいトピックを代入
                self.z_k[d][n] = z_new
                self.K_V[z_new][w] += 1
                self.D_K[d][z_new] += 1
                if bool_:
                    self.K_C_V[z_new][const_id][w] += 1

    # トピック分布の表示
    def get_topic(self):
        return (self.D_K + self.alpha) / np.sum((self.D_K + self.alpha), axis=1)[:, np.newaxis]

    # 制約のパラメータの更新
    def set_const_params(self):
        self.K_C_V = np.zeros(shape=(self.K, self.C, self.V))
        for d, (words, zs) in enumerate(zip(self.bw, self.z_k)):
            for w, z in zip(words, zs):
                bool_, const_id = self.search_const(w)
                if bool_:
                    self.K_C_V[z][const_id][w] += 1

    # 制約の追加
    def set_const(self, const_new = []):
        for const_one in const_new:
            w1, w2 = const_one
            bool1, const_id1 = self.search_const(w1)
            bool2, const_id2 = self.search_const(w2)
            if all([bool1, bool2]):
                if const_id1 == const_id2:
                    pass
                else:
                    self.const.append(self.const[const_id1]+self.const[const_id2])
                    if const_id2 > const_id1:
                        del self.const[const_id2]
                        del self.const[const_id1]
                    else:
                        del self.const[const_id1]
                        del self.const[const_id2]
            elif bool1:
                self.const[const_id1].append(w2)
            elif bool2:
                self.const[const_id2].append(w1)
            else:
                self.const.append([w1, w2])
        self.C = len(self.const)
        self.set_const_params()

    # 単語分布の表示
    def get_word(self):
        return (self.K_V + self.beta) / np.sum((self.K_V + self.beta), axis=1)[:, np.newaxis]


def main():
    # 9つの文章に対して、９つの単語、３つのトピック
    bw = np.array([[0, 1, 2],
                   [0, 1, 2],
                   [0, 1, 2, 2],
                   [3, 4, 5],
                   [3, 3, 4, 5],
                   [3, 4, 4, 5],
                   [6, 7, 8],
                   [6, 7, 7, 8],
                   [7, 7, 8, 8]])
    K = 3
    itm = ITM(bw=bw, K=K)
    itm.set_params()
    for iter_ in range(100):
        itm.infer_z()

    print "===単語分布==="
    print itm.get_word()
    print "===トピック分布==="
    print itm.get_topic()

    # 制約の追加
    itm.set_const(const_new=[[2,3], [1,4]])
    print itm.K_C_V
    for iter_ in range(100):
        itm.infer_z()

    print "===単語分布==="
    print itm.get_word()
    print "===トピック分布==="
    print itm.get_topic()


if __name__ == '__main__':
    main()
