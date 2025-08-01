import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Own_LR:
    def __init__(self, learning_rate=0.01, max_iters=100, degree=1, n_class=3):
        """
        learning_rate -- 学习率 (默认: 0.01)
        max_iters -- 迭代次数 (默认: 1000)
        degree -- 多项式特征的次数 (默认: 1)
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.degree = degree
        self.n_class = n_class
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.X_mean = None
        self.X_std = None
        
    # 标准化数据
    def datastandard(self, X):   
        # 标准化数据
        if self.X_mean is None:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            # 避免除以0
            self.X_std[self.X_std == 0] = 1
        return (X - self.X_mean) / self.X_std
    
    # 多项式特征生成
    def poly_own(self, X):
        n_samples, n_features = X.shape
        features = [np.ones((n_samples, 1))]  # 添加偏置项
        
        for j in range(1, self.degree + 1):
            for i in range(n_features):
                features.append(X[:, i:i+1] ** j)   # 得到每个特征（第i列）的每个次幂，加入到features列表中
                
        return np.hstack(features)
    
    def proba(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        # 标准化
        X = self.datastandard(X)
        
        # 添加多项式特征
        X_poly = self.poly_own(X)
        
        n_samples, n_features = X_poly.shape
        
        # 初始化参数
        self.weights = np.zeros((n_features, self.n_class))
        self.bias = np.zeros(self.n_class)
        
        # 梯度下降
        for i in range(self.max_iters):
            # 计算预测值
            linear_model = np.dot(X_poly, self.weights) + self.bias
            y_pred = self.proba(linear_model)
            
            # 计算损失 
            loss = (-1/n_samples) * np.sum(y * np.log(y_pred))
            self.loss_history.append(loss)
            
            # 计算梯度
            error = y_pred - y
            dw = (1/n_samples) * np.dot(X_poly.T, error)
            db = (1/n_samples) * np.sum(error, axis=0)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict_proba(self, X):
        X = self.datastandard(X)
        X_poly = self.poly_own(X)
        linear_model = np.dot(X_poly, self.weights) + self.bias
        return self.proba(linear_model)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    

if __name__ == "__main__":
    path = './DAY1/iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes
    x, y = np.split(data.values, (4,), axis=1)
    # 仅使用前两列特征
    x = x[:, :2]

    
    # 创建并训练模型
    model = Own_LR(learning_rate=0.1, max_iters=1000, degree=1, n_class=3)
    model.fit(x, y)
    
    # 进行预测
    y_hat = model.predict(x)
    y_hat_prob = model.predict_proba(x)
    
    np.set_printoptions(suppress=True)
    # print('y_hat = \n', y_hat)
    # print('y_hat_prob = \n', y_hat_prob)
    print('准确度：%.2f%%' % (100 * np.mean(y_hat == y.ravel())))
    # 画图
    N, M = 200, 200  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = model.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y.flat, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    plt.xlabel(u'花萼长度', fontsize=14)
    plt.ylabel(u'花萼宽度', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
    plt.legend(handles=patchs, fancybox=True, framealpha=0.8)
    plt.title(u'鸢尾花Logistic回归分类效果 - 标准化', fontsize=17)
    plt.show()
