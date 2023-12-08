from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    '''
    激活函数
    '''
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class BPClassification:
    def __init__(self, sizes):

        # 神经网络结构
        self.num_layers = len(sizes)
        self.sizes = sizes

        # 初始化偏差，除输入层外， 其它每层每个节点都生成一个 biase 值（0-1）
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]
        # 随机生成每条神经元连接的 weight 值（0-1）
        self.weights = [np.random.randn(r, c)
                        for c, r in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        '''
        前向传输计算输出神经元的值
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def MSGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        小批量随机梯度下降法
        '''
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in np.arange(epochs):
            # 随机打乱训练集顺序
            random.shuffle(training_data)
            # 根据小样本大小划分子训练集集合
            mini_batchs = [training_data[k:k + mini_batch_size]
                           for k in np.arange(0, n, mini_batch_size)]
            # 利用每一个小样本训练集更新 w 和 b
            for mini_batch in mini_batchs:
                self.updata_WB_by_mini_batch(mini_batch, eta)

            # 迭代一次后结果
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0}".format(j))

    def updata_WB_by_mini_batch(self, mini_batch, eta):
        '''
        利用小样本训练集更新 w 和 b
        mini_batch: 小样本训练集
        eta: 学习率
        '''
        # 创建存储迭代小样本得到的 b 和 w 偏导数空矩阵，大小与 biases 和 weights 一致，初始值为 0
        batch_par_b = [np.zeros(b.shape) for b in self.biases]
        batch_par_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 根据小样本中每个样本的输入 x, 输出 y, 计算 w 和 b 的偏导
            delta_b, delta_w = self.back_propagation(x, y)
            # 累加偏导 delta_b, delta_w
            batch_par_b = [bb + dbb for bb, dbb in zip(batch_par_b, delta_b)]
            batch_par_w = [bw + dbw for bw, dbw in zip(batch_par_w, delta_w)]
        # 根据累加的偏导值 delta_b, delta_w 更新 b, w
        # 由于用了小样本，因此 eta 需除以小样本长度
        self.weights = [w - (eta / len(mini_batch)) * dw
                        for w, dw in zip(self.weights, batch_par_w)]
        self.biases = [b - (eta / len(mini_batch)) * db
                       for b, db in zip(self.biases, batch_par_b)]

    def back_propagation(self, x, y):
        '''
        利用误差后向传播算法对每个样本求解其 w 和 b 的更新量
        x: 输入神经元，行向量
        y: 输出神经元，行向量
        '''
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播，求得输出神经元的值
        a = x  # 神经元输出值
        # 存储每个神经元输出
        activations = [x]
        # 存储经过 sigmoid 函数计算的神经元的输入值，输入神经元除外
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)  # 输出神经元
            activations.append(a)

        # 求解输出层δ
        delta = self.cost_function(activations[-1], y) * sigmoid_prime(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].T)
        for lev in np.arange(2, self.num_layers):
            # 从倒数第1层开始更新，因此需要采用-lev
            # 利用 lev + 1 层的 δ 计算 l 层的 δ
            z = zs[-lev]
            zp = sigmoid_prime(z)
            delta = np.dot(self.weights[-lev + 1].T, delta) * zp
            delta_b[-lev] = delta
            delta_w[-lev] = np.dot(delta, activations[-lev - 1].T)
        return (delta_b, delta_w)

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feed_forward(x)), y)
                       for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    def predict(self, test_input):
        test_result = [self.feed_forward(x)
                       for x in test_input]
        return test_result

    def cost_function(self, output_a, y):
        '''
        损失函数
        '''
        return (output_a - y)

    # 以下是几种常见评价指标
    def accuracy(self,x_test,y_test):
        y_predict = np.array([ np.argmax(i) for i in self.predict(x_test)]).astype(y_test.dtype)
        return np.sum(y_test==y_predict)/y_test.shape[0]

    def macro_precision(self,x_test,y_test):
        y_unique = np.unique(y_test)
        y_predict = np.array([ np.argmax(i) for i in self.predict(x_test)]).astype(y_test.dtype)
        P = []
        for i in y_unique:
            P.append(np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_predict==i))
        return np.sum(P)/len(P)

    def macro_recall(self,x_test,y_test):
        y_unique = np.unique(y_test)
        y_predict = np.array([ np.argmax(i) for i in self.predict(x_test)]).astype(y_test.dtype)
        R = []
        for i in y_unique:
            R.append(np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_test==i))
        return np.sum(R) / len(R)

    def macro_f1_score(self,x_test,y_test):
        y_unique = np.unique(y_test)
        y_predict = np.array([ np.argmax(i) for i in self.predict(x_test)]).astype(y_test.dtype)
        F = []
        for i in y_unique:
            p = np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_predict==i)
            r = np.sum(y_predict[y_predict==y_test]==i)/np.sum(y_test==i)
            F.append((2*p*r)/(p+r))
        return np.sum(F) / len(F)


    def precision(self,x_test,y_test):
        y_predict = np.array([ np.argmax(i) for i in self.predict(x_test)]).astype(y_test.dtype)
        y_unique = np.unique(y_test)
        return np.sum(y_predict[y_predict==y_test]==y_unique[0])/np.sum(y_predict==y_unique[0]),np.sum(y_predict[y_predict==y_test]==y_unique[1])/np.sum(y_predict==y_unique[1])
    def recall(self,x_test,y_test):
        y_predict = np.array([ np.argmax(i) for i in self.predict(x_test)]).astype(y_test.dtype)
        y_unique = np.unique(y_test)
        return np.sum(y_predict[y_predict==y_test]==y_unique[0])/np.sum(y_test==y_unique[0]),np.sum(y_predict[y_predict==y_test]==y_unique[1])/np.sum(y_test==y_unique[1])
    def f1_score(self,x_test,y_test):
        precision = self.precision(x_test,y_test)
        recall = self.recall(x_test,y_test)
        return 2*precision[0]*recall[0]/(precision[0]+recall[0]),2*precision[1]*recall[1]/(precision[1]+recall[1])
    def report(self,x_test,y_test,labels):
        y_predict = np.array([ np.argmax(i) for i in self.predict(x_test)]).astype(y_test.dtype)
        report = classification_report(y_test,y_predict,labels=labels)
        return report

if __name__=='__main__':
    # ///////////////////////////////////////////////// Iris
    data = pd.read_csv('data/D2.csv').drop(['Unnamed: 0'],axis = 1)
    X = np.array(data.drop(columns=['Species']))
    y = np.array(data['Species'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    print('IRIS_BP神经网络下------------------------------------------')
    train_x = [i.reshape(-1,1) for i in X_train]
    train_y = [np.array([[1 if y==0 else 0],
                           [1 if y==1 else 0],
                           [1 if y==2 else 0]])
                           for y in y_train]
    train_data = [[x, y] for x, y in zip(train_x, train_y)]
    test_x = [i.reshape(-1,1) for i in X_test]
    test_data = [[x, y] for x, y in zip(test_x, y_test)]
    bp = BPClassification([4, 15, 3])
    bp.MSGD(train_data, 1000, 10, 0.5)
    print('训练集下：------------------------------------')
    print(bp.report(train_x, y_train, labels=[0, 1,2]))
    print('测试集下：------------------------------------')
    print('accuracy:', bp.accuracy(test_x, y_test))
    print('macro_precision:', bp.macro_precision(test_x, y_test))
    print('macro_recall:', bp.macro_recall(test_x, y_test))
    print('macro_f1_score:', bp.macro_f1_score(test_x, y_test))
    print('micro版本==accuracy:', bp.accuracy(test_x, y_test))
    print(bp.report(test_x, y_test, labels=[0, 1,2]))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots()
    C5 = confusion_matrix(y_test, np.array([ np.argmax(i) for i in bp.predict(test_x)]).astype(y_test.dtype), labels=[0,1,2])
    sns.heatmap(C5, annot=True, ax=ax)  # 画热力图
    ax.set_title('神经网络混淆矩阵')  # 标题
    ax.set_xlabel('预测值')  # x轴
    ax.set_ylabel('真实值')  # y轴
    plt.show()
    # ///////////////////////////////////////////////// Wine
    data1 = pd.read_csv('data/D3.csv').drop(['Unnamed: 0'], axis=1)
    X1 = np.array(data1.drop(columns=['quality']))
    y1 = np.array(data1['quality'])
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=3)

    # ----------------- BP
    print('Wine_BP神经网络下------------------------------------------')
    train_x1 = [i.reshape(-1,1) for i in X_train1]
    train_y1 = [np.array([[1 if y==0 else 0],
                           [1 if y==1 else 0]])
                           for y in y_train1]
    train_data1 = [[x, y] for x, y in zip(train_x1, train_y1)]
    test_x1 = [i.reshape(-1,1) for i in X_test1]
    test_data1 = [[x, y] for x, y in zip(test_x1, y_test1)]
    bp1 = BPClassification([11, 15, 2])
    bp1.MSGD(train_data1, 1000, 10, 0.5)
    print('训练集下：------------------------------------')
    print(bp1.report(train_x1, y_train1, labels=[0, 1]))
    print('测试集下：------------------------------------')
    print('accuracy:', bp1.accuracy(test_x1, y_test1))
    print('macro_precision:', bp1.macro_precision(test_x1, y_test1))
    print('macro_recall:', bp1.macro_recall(test_x1, y_test1))
    print('macro_f1_score:', bp1.macro_f1_score(test_x1, y_test1))
    print('micro版本==accuracy:', bp1.accuracy(test_x1, y_test1))
    print(bp1.report(test_x1, y_test1, labels=[0, 1]))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots()
    C6 = confusion_matrix(y_test1, np.array([ np.argmax(i) for i in bp1.predict(test_x1)]).astype(y_test1.dtype), labels=[0,1])
    sns.heatmap(C6, fmt='.0f',annot=True, ax=ax)  # 画热力图
    ax.set_title('神经网络混淆矩阵')  # 标题
    ax.set_xlabel('预测值')  # x轴
    ax.set_ylabel('真实值')  # y轴
    plt.show()
