# coding: utf-8
"""
非負値行列因子分解をするためのサンプル
"""
import numpy as np


class NMF:
    def __init__(self, X, K):
        self.X = X
        self.I = X.shape[0]
        self.J = X.shape[1]
        self.K = K
        self.T = np.absolute(np.random.randn(self.I, K))
        self.V = np.absolute(np.random.randn(K, self.J))

    def distance_EU(self):
        for i in range(self.I):
            for k in range(self.K):
                nume = np.sum(self.X[i] * self.V[k])
                deno = np.sum(self.T.dot(self.V)[i] * self.V[k])
                self.T[i][k] *= nume / deno
        for k in range(self.K):
            for j in range(self.J):
                nume = np.sum(self.X[:, j] * self.T[:, k])
                deno = np.sum(self.T.dot(self.V)[:, j] * self.T[:, k])
                self.V[k][j] *= nume / deno

    def distance_KL(self):
        for i in range(self.I):
            for k in range(self.K):
                nume = np.sum(self.X[i] / self.T.dot(self.V)[i] * self.V[k])
                deno = np.sum(self.V[k])
                self.T[i][k] *= nume / deno
        for k in range(self.K):
            for j in range(self.J):
                nume = np.sum(self.X[:, j] / self.T.dot(self.V)[:, j] * self.T[:, k])
                deno = np.sum(self.T[:, k])
                self.V[k][j] *= nume / deno

    def distance_IS(self):
        for i in range(self.I):
            for k in range(self.K):
                _X = self.T.dot(self.V)
                nume = np.sum(self.X[i] * self.V[k] / _X[i] / _X[i])
                deno = np.sum(self.V[k] / _X[i])
                self.T[i][k] *= np.sqrt(nume / deno)
        for k in range(self.K):
            for j in range(self.J):
                _X = self.T.dot(self.V)
                nume = np.sum(self.X[:, j] * self.T[:, k] / _X[:, j] / _X[:, j])
                deno = np.sum(self.T[:, k] / _X[:, j])
                self.V[k][j] *= np.sqrt(nume / deno)


X1 = np.array([[1, 1, 2, 3, 1],
              [0, 1, 0, 1, 1],
              [2, 0, 4, 4, 0],
              [3, 0, 6, 6, 0]])


X2 = np.array([[1,1,1,0,0,0,0,0],
               [1,2,1,0,0,0,0,0],
               [1,1,2,0,0,0,0,0],
               [0,0,0,1,1,1,0,0],
               [0,0,0,2,1,1,0,0],
               [0,0,0,2,2,1,0,0],
               [0,0,0,0,0,0,1,1],
               [0,0,0,0,0,0,1,1],
               [0,0,0,0,0,0,2,2]])

nmf = NMF(X2, 3)
for i in range(5):
    nmf.distance_KL()
print nmf.V
print ""
print nmf.T.dot(nmf.V)
