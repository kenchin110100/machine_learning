# coding: utf-8
"""
EMアルゴリズムを用いて混合直線回帰を解くコード
"""
import numpy as np
import matplotlib.pyplot as plt


class EM:
    def __init__(self, data):
        np.random.shuffle(data)
        self.data = data
        self.group1 = data[0:data.shape[0]//2]
        self.group2 = data[data.shape[0]//2+1:data.shape[0]]
        self.z1 = np.array([[0, 0]])
        self.z2 = np.array([[0, 0]])

    def estep(self):
        A1 = self.group1[:, 0:2]
        y1 = self.group1[:, 2:3]
        A2 = self.group2[:, 0:2]
        y2 = self.group2[:, 2:3]
        self.z1 = np.linalg.inv(A1.T.dot(A1)).dot(A1.T).dot(y1)
        self.z2 = np.linalg.inv(A2.T.dot(A2)).dot(A2.T).dot(y2)

    def mstep(self):
        group1_tmp = []
        group2_tmp = []
        for row in self.data:
            distance1 = np.abs(self.z1[0][0]*row[0] + self.z1[1][0] - row[2]) / np.sqrt(self.z1[0][0] ** 2 + self.z1[1][0] ** 2)
            distance2 = np.abs(self.z2[0][0]*row[0] + self.z2[1][0] - row[2]) / np.sqrt(self.z2[0][0] ** 2 + self.z2[1][0] ** 2)
            if distance1 < distance2:
                group1_tmp.append(row)
            else:
                group2_tmp.append(row)
        self.group1 = np.array(group1_tmp)
        self.group2 = np.array(group2_tmp)


data1_x = np.random.randint(0, 50, 50)
data2_x = np.random.randint(0, 50, 50)
data1_y = data1_x * 2 - 10 + np.random.randn(50) * 3
data2_y = data2_x * 0.5 + 20 + np.random.randn(50) * 3
data1 = np.array([data1_x, np.ones(50), data1_y])
data2 = np.array([data2_x, np.ones(50), data2_y])
data = np.c_[data1, data2]
data = data.T
em = EM(data)
for i in range(3000):
    em.estep()
    em.mstep()
print '###z1###'
print em.z1
print '###z2###'
print em.z2

plot_x = np.c_[np.arange(50), np.ones(50)].T

plt.plot(data[:, 0:1], data[:, 2:3], '.')
plt.plot(plot_x[0:1, :][0], em.z1.T.dot(plot_x)[0], '-')
plt.plot(plot_x[0:1, :][0], em.z2.T.dot(plot_x)[0], '-')
plt.show()
