#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
import pickle

def read_data(filename):
    # 读取数据集
    datafile = open('housing.data')
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    data = []
    for line in datafile:
        data.append(list(map(float, line.split())))

    # 均一化
    cols = [[one[i] for one in data] for i in range(feature_num)]
    maximums = []
    minimums = []
    avg = []

    for i in range(feature_num):
        maximums.append(max(cols[i]))
        minimums.append(min(cols[i]))
        avg.append(sum(cols[i]) / len(cols[i]))

    for one in data:
        for i in range(feature_num):
            # print(maximums[i], minimums[i], avgs[i])
            one[i] = (one[i] - avg[i]) / (maximums[i] - minimums[i])

    # 分割数据集，分为训练集和测试集
    ratio = 0.8
    training_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    return training_data, feature_num, avg, maximums, minimums


class Network(object):

    def __init__(self, num_of_weights):
        self.W_1 = []
        self.W_2 = []
        self.B_1 = []
        self.B_2 = (random.uniform(0, 2.0)-1)
        for i in range(feature_num - 1):
            w = []
            for j in range(feature_num - 1):
                w.append((random.uniform(0, 2.0)-1))
            self.W_1.append(w)
            self.W_2.append((random.uniform(0, 2.0)-1))
            self.B_1.append((random.uniform(0, 2.0)-1))
        self.eta = 0.1

    def load(self):
        with open('save.data', 'rb') as savefile:
            save=pickle.load(savefile)
            self.W_1=save[0]
            self.W_2=save[1]
            self.B_1=save[2]
            self.B_2=save[3]

    def forward(self, data_pack):
        Z_1_result = []
        Z_2_result = []
        for one in data_pack:
            z = 0
            Z_1 = []
            Z_2 = 0
            for i in range(feature_num - 1):
                for j in range(feature_num - 1):
                    z += self.W_1[i][j] * one[j]
                z += self.B_1[i]
                Z_1.append(z)
                z = 0
            Z_1_result.append(Z_1)
            for i in range(feature_num - 1):
                Z_2 += Z_1[i]*self.W_2[i]
            Z_2 += self.B_2
            Z_2_result.append(Z_2)
        return Z_1_result, Z_2_result

    def loss(self, z, real): # 计算损失函数
        cost = []
        for i in range(len(z)):
            error = z[i] - real[i]
            costone = error**2
            cost.append(costone)
            pass

        res = sum(cost) / len(z)
        return res

    def gradient(self, x, real, z1, z2): # 计算梯度
        gradient_W_1 = []
        gradient_W_2 = []
        gradient_B_1 = []
        gradient_B_2 = []

        for n in range(len(x)):

            tmp_g = (z2[n]-real[n])
            gradient_B_2.append(tmp_g)

            gradient_w2 = []
            for i in range(feature_num - 1):
                gradient_w2.append(tmp_g * z1[n][i])
            gradient_W_2.append(gradient_w2)

            gradient_b1 = []
            for i in range(feature_num - 1):
                gradient_b1.append(tmp_g * self.W_2[i])
            gradient_B_1.append(gradient_b1)

            gradient_w1 = []
            for i in range(feature_num - 1):
                g_line_w1 = []
                for j in range(feature_num - 1):
                    g_line_w1.append(tmp_g * self.W_2[i] * x[n][j])
                gradient_w1.append(g_line_w1)
            gradient_W_1.append(gradient_w1)
        #求和取平均
        cols_W_2 = [[one[i] for one in gradient_W_2] for i in range(feature_num - 1)]
        cols_B_1 = [[one[i] for one in gradient_B_1] for i in range(feature_num - 1)]
        cols_W_1 = [[[one[i][j] for one in gradient_W_1]for j in range(feature_num - 1)] for i in range(feature_num -1)]
        aver_W_2 = []
        aver_B_1 = []
        aver_W_1 = []
        for i in range(feature_num - 1):
            aver_W_2.append(sum(cols_W_2[i]) / len(cols_W_2[i]))
            aver_B_1.append(sum(cols_B_1[i]) / len(cols_B_1[i]))
            aver_w1 = []
            for j in range(feature_num - 1):
                aver_w1.append(sum(cols_W_1[i][j]) / len(cols_W_1[i][j]))
            aver_W_1.append(aver_w1)
        aver_B_2 = sum(gradient_B_2) / len(gradient_B_2)

        return aver_W_1, aver_W_2, aver_B_1, aver_B_2

    def update(self, gradient_w1, gradient_w2, gradient_b1, gradient_b2, eta=0.1):
        self.B_2 = self.B_2 - eta * gradient_b2
        for i in range(feature_num - 1):
            self.B_1[i] = self.B_1[i] - eta * gradient_b1[i]
            self.W_2[i] = self.W_2[i] - eta * gradient_w2[i]
            for j in range(feature_num - 1):
                self.W_1[i][j] = self.W_1[i][j] - eta * gradient_w1[i][j]

    def train(self, training_data, num_epoches=100, batch_size=100):
        num = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱，
            random.shuffle(training_data)
            x_all = [i[:-1] for i in training_data]
            y_all = [i[-1:][0] for i in training_data]

            for i in range(0, num, batch_size):
                x = x_all[i:i + batch_size]
                y = y_all[i:i + batch_size]
                z1, z2 = self.forward(x)
                loss = self.loss(z2, y)
                gradient_w1, gradient_w2, gradient_b1, gradient_b2 = self.gradient(x, y, z1, z2)
                self.update(gradient_w1, gradient_w2, gradient_b1, gradient_b2, self.eta)
                losses.append(loss)
                print('Epoch {:3d} , loss = {:.4f}'.
                      format(epoch_id, loss))
        save = []
        save.append(self.W_1)
        save.append(self.W_2)
        save.append(self.B_1)
        save.append(self.B_2)
        with open('save.data', 'wb') as savefile:
            pickle.dump(save, savefile)

        return losses


if __name__ == "__main__":
    # 读取数据集
    training_data, feature_num, avg, maximums, minimums = read_data('./housing.data')
    # 启动训练
    net = Network(13)
    key=input("请选择\n1 直接训练\n2 使用已有的模型\n")
    if key == '2' :
        net.load()
    else:
        losses = net.train(training_data, num_epoches=100)
        plot_x = range(len(losses))
        plot_y = losses
        plt.plot(plot_x, plot_y)
        plt.show()
    new = input("训练完毕，请输入要预测的数据\n")
    new_pack = []
    new_pack.append(list(map(float, new.split())))
    for i in range(feature_num - 1):
        # print(maximums[i], minimums[i], avg[i])
        new_pack[0][i] = (new_pack[0][i] - avg[i]) / (maximums[i] - minimums[i])
    newz1, newz2 = net.forward(new_pack)
    # 将归一化的数据复原
    newz2[0] = newz2[0]*(maximums[feature_num-1] - minimums[feature_num-1])+avg[feature_num-1]
    print("预测结果为")
    print(newz2)
