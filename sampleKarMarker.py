# coding: utf-8
"""
カーマーカー法によって線形計画問題を解くためのサンプル
(正確にはアフィン・スケーリング法)
"""
import numpy as np
import copy
import matplotlib.pyplot as plt


def karmarker_one(x, c, A, b, gamma=1.0, eps=0.001):
    """
    x: 解，縦ベクトル
    c: コスト，縦ベクトル
    A: 制約条件の係数，マトリックス
    b: 制約条件の上限，縦ベクトル
    """
    vk = b - A.dot(x)
    D_v_2 = np.diag(1.0/np.square(vk.T[0]))
    h_x = np.linalg.pinv(A.T.dot(D_v_2).dot(A)).dot(c)
    if np.linalg.norm(h_x) < eps:
        return x, 1
    h_v = -1*A.dot(h_x)
    if np.min(h_v) >= 0:
        return None, 1

    alpha = gamma*np.min([-1*v/h for v, h in zip(vk.T[0], h_v.T[0]) if h < 0])
    x += alpha * h_x
    return x, 0


def main(c, A, b, gamma=1.0, eps=0.001):
    """
    object max z = c^T * x
    subject Ax <= b
    """
    # 戻り値
    value = 0
    x = np.array([1.0 for i in range(len(A[0]))])[:,np.newaxis]
    list_x = [copy.deepcopy(x).T[0]]
    while value == 0:
        x, value = karmarker_one(x, c, A, b, gamma=gamma, eps=eps)
        list_x.append(copy.deepcopy(x).T[0])
    if x == None:
        print "Unbonded!!"
    else:
        return list_x


if __name__ == "__main__":
    c = np.array([2.0, 5.0], dtype="float32")[:,np.newaxis]
    A = np.array([[2.0, 6.0],
                  [8.0, 6.0],
                  [3.0, 1.0],
                  [-1.0, 0.0],
                  [0.0, -1.0]], dtype="float32")
    b = np.array([27.0, 45.0, 15.0, 0.0, 0.0], dtype="float32")[:,np.newaxis]
    list_x = main(c, A, b, gamma=0.1, eps=0.001)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    subject_x1 = [i/100.0 for i in range(0, 320)]
    subject_x2 = [i/100.0 for i in range(280, 470)]
    subject_x3 = [i/100.0 for i in range(430, 500)]
    subject_y1 = [27/6.0-x/3.0 for x in subject_x1]
    subject_y2 = [45/6.0-4*x/3.0 for x in subject_x2]
    subject_y3 = [15.0-3*x for x in subject_x3]
    arr_x = np.array(list_x).T[0]
    arr_y = np.array(list_x).T[1]
    ax.plot(arr_x, arr_y, "o")
    ax.plot(subject_x1, subject_y1, lw=2)
    ax.plot(subject_x2, subject_y2, lw=2)
    ax.plot(subject_x3, subject_y3, lw=2)
    ax.grid(True)
    plt.show()

