# coding: utf-8
"""
マルコフ連鎖をシミュレーションするサンプル
"""
import numpy as np

# 推移行列
A = np.array([[1,1,1,1,1], [4,3,5,1,2], [2,3,5,1,4], [3,4,2,5,1], [5,4,1,2,3]], dtype=float)

def normalize(A):
    G = np.array([row/np.sum(row) for row in A])
    return G

def main():
    # 推移確率行列に変換
    G = normalize(A)
    R = np.array([float(1)/len(G) for i in range(len(G))], dtype=float).T
    increment = 1
    while increment > 0.00001:
        R_rev = G.dot(R)
        increment = np.sum(np.abs(R_rev - R))
        print R_rev
        R = R_rev

    print R


if __name__ == '__main__':
    main()
