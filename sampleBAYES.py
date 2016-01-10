# coding: utf-8
"""
ベイズフィッティングのサンプル
"""
import numpy as np
from pylab import *

# M次多項式近似
M = 3
ALPHA = 0.005
BETA = 11.1


# xの特徴ベクトル、(サンプル数, M+1)の行列を作成
def cal_arr_X(arr_x):
    arr_X = np.ones((len(arr_x)))
    for i in range(1, M+1):
        arr_X = np.c_[arr_X, arr_x ** i]
    return arr_X


# 行列Sを作成する
def cal_arr_S(arr_X):
    size = arr_X.shape[1]
    arr_S_inv = ALPHA * np.identity(size) + BETA * arr_X.T.dot(arr_X)
    arr_S = np.linalg.inv(arr_S_inv)
    return arr_S


# muを計算する
def cal_mu(x, arr_x, arr_t):
    arr_new_x = np.array([x ** i for i in range(0, M+1)])
    arr_X = cal_arr_X(arr_x)
    arr_S = cal_arr_S(arr_X)
    arr_sum = np.sum(arr_X * arr_t.reshape(len(arr_t), 1), axis=0)

    mu_x = BETA * arr_new_x.dot(arr_S).dot(arr_sum.reshape(len(arr_sum), 1))

    return mu_x


# sigmaを計算する
def cal_sigma(x, arr_x, arr_t):
    arr_new_x = np.array([x ** i for i in range(0, M+1)])
    arr_X = cal_arr_X(arr_x)
    arr_S = cal_arr_S(arr_X)
    sigma_x = 1.0 / BETA + arr_new_x.dot(arr_S).dot(arr_new_x.reshape(len(arr_new_x), 1))

    return sigma_x


def main():
    # 訓練データ
    # sin(2*pi*x)の関数値にガウス分布に従う小さなランダムノイズを加える
    arr_x = np.linspace(0, 1, 10)
    arr_t = np.sin(2*np.pi*arr_x) + np.random.normal(0, 0.2, arr_x.size)

    # 連続関数のプロット用X値
    xs = np.linspace(0, 1, 500)
    ideal = np.sin(2*np.pi*xs)         # 理想曲線
    means = np.array([cal_mu(x, arr_x, arr_t) for x in xs])
    sigmas = np.array([np.sqrt(cal_sigma(x, arr_x, arr_t)) for x in xs])
    uppers = means + sigmas
    lowers = means - sigmas

    plot(arr_x, arr_t, 'bo')  # 訓練データ
    plot(xs, ideal, 'g-')     # 理想曲線
    plot(xs, means, 'r-')     # 予測モデルの平均
    plot(xs, uppers, 'r--')   # +sigma
    plot(xs, lowers, 'r--')   # -sigma
    xlim(0.0, 1.0)
    ylim(-1.5, 1.5)
    show()

if __name__ == "__main__":
    main()
