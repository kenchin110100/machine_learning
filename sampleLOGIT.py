# coding: utf-8
"""
ロジスティック回帰をするためのサンプル
ガウス分布を仮定して、解析的に解く方法と
識別関数として解く方法の比較
"""
import numpy as np
import matplotlib.pyplot as plt


#平均
mu1 = np.array([-4,-4])
mu2 = np.array([4,4])
mu3 = np.array([-4,4])
#共分散
cov1 = np.array([[1,0], [0,1]])
cov2 = np.array([[1,0], [0,1]])

A = np.random.multivariate_normal(mu1,cov1,100)
B1 = np.random.multivariate_normal(mu2,cov2,50)
B2 = np.random.multivariate_normal(mu3,cov2,50)
B = np.r_[B1, B2]
print B

def gen_cal_w(A, B, cov1, cov2, C1=1, C2=1):
    muA = np.average(A.T, axis=1).T
    muB = np.average(B.T, axis=1).T
    cov = np.float(C1)/(C1+C2)*cov1 + np.float(C2)/(C1+C2)*cov2
    inv_cov = np.linalg.inv(cov)
    w = inv_cov.dot(muA - muB)
    w0 = -1.0/2*muA.T.dot(inv_cov).dot(muA) + 1.0/2*muB.T.dot(inv_cov).dot(muB)+np.log(C1/C2)
    return w, w0

def logit(x):
    return 1.0/(1.0+np.exp(-1*x))[0]

def cls_cal_w(A, B, cov1, cov2, C1, C2):
    phi_tmp = np.r_[A, B]
    phi = np.c_[np.ones((C1+C2, 1)), phi_tmp]
    t = np.r_[np.ones((C1,1)), np.zeros((C2,1))]
    w_old = np.random.rand(len(phi[0]), 1)
    converge = 1
    while converge > 0.000000000001:
        R = np.diag([logit(row.dot(w_old))*(1-logit(row.dot(w_old))) for row in phi])
        # ヘッセ行列
        H = phi.T.dot(R).dot(phi)
        # yベクトル
        y = np.array([[logit(row.dot(w_old))] for row in phi])
        z = phi.dot(w_old) - np.linalg.inv(R).dot(y-t)
        w_new = np.linalg.inv(H).dot(phi.T).dot(R).dot(z)
        converge = np.sum(w_new-w_old)
        w_old = w_new

    return w_new


def main():
    # 解析的な答えで
    gen_w, gen_w0 = gen_cal_w(A, B, cov1, cov2, len(A), len(B))
    xs = np.linspace(-5, 5, 500)
    gen_ys = -1*gen_w.T[0]/gen_w.T[1]*xs - gen_w0/gen_w.T[1]

    # 反復計算法で
    cls_w = cls_cal_w(A, B, cov1, cov2, len(A), len(B))
    print cls_w
    cls_ys = -1*cls_w[1][0]/cls_w[2][0]*xs - cls_w[0][0]/cls_w[2][0]

    # プロット
    plt.plot(A.T[0], A.T[1], 'bo')
    plt.plot(B.T[0], B.T[1], 'bo', color='r')
    plt.plot(xs, gen_ys, color='g')
    plt.plot(xs, cls_ys, color='y')
    plt.show()


if __name__ == "__main__":
    main()
