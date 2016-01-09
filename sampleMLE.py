# coding: utf-8
"""
最尤推定のサンプルコード
"""
import numpy as np
from pylab import *

# M次多項式近似
M = 3

def make_arr_X(arr_x):
    arr_X = np.ones((len(arr_x)))
    for i in range(1, M+1):
        arr_X = np.c_[arr_X, arr_x ** i]
    return arr_X


# 訓練データからパラメータを推定
def estimate(arr_x, arr_t):
    arr_X = make_arr_X(arr_x)
    arr_X_X = np.dot(arr_X.T, arr_X)
    arr_X_X_1 = np.linalg.inv(arr_X_X)
    arr_w = arr_X_X_1.dot(arr_X.T).dot(arr_t)
    arr_y = arr_w.dot(arr_X.T)
    sigma = np.sum((arr_t - arr_y) ** 2) / len(arr_x)
    return arr_w, sigma

def main():
    # 訓練データ
    # sin(2*pi*x)の関数値にガウス分布に従う小さなランダムノイズを加える
    arr_x = np.linspace(0, 1, 10)
    arr_t = np.sin(2*np.pi*arr_x) + np.random.normal(0, 0.2, arr_x.size)

    # 訓練データからパラメータw_mlを推定
    arr_w, sigma = estimate(arr_x, arr_t)
    print arr_w

    # 連続関数のプロット用X値
    xs = np.linspace(0, 1, 500)
    ideal = np.sin(2*np.pi*xs)         # 理想曲線
    means = []
    arr_X = make_arr_X(xs)
    means = arr_w.dot(arr_X.T)
    s = sqrt(sigma)  # 予測分布の標準偏差
    uppers = means + s          # 平均 + 標準偏差
    lowers = means - s          # 平均 - 標準偏差

    plot(arr_x, arr_t, 'bo')  # 訓練データ
    plot(xs, ideal, 'g-')     # 理想曲線
    plot(xs, means, 'r-')     # 予測モデルの平均
    plot(xs, uppers, 'r--')
    plot(xs, lowers, 'r--')
    xlim(0.0, 1.0)
    ylim(-1.5, 1.5)
    show()

if __name__ == "__main__":
    main()
