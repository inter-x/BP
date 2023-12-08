import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
def Z_score(data):
    '''Z-score标准化过程'''
    Stand = StandardScaler()
    D1_zscore = pd.DataFrame(Stand.fit_transform(data),
                             columns=data.columns)
    return D1_zscore
def describe(data,num):
    sns.set(font='SimHei', font_scale=1.2)
    desc = data.describe().T  # 对于count mean std min 25% 50% 75% max 进行展示
    desc_df = pd.DataFrame(index=data.columns, columns=data.describe().index, data=desc)
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(desc_df, annot=True, cmap="Purples", fmt='.3f',
                ax=ax, linewidths=5, cbar=False,
                annot_kws={"size": 16})  # 绘制热力图
    plt.xticks(size=18)
    plt.yticks(size=14, rotation=0)
    plt.savefig('图片/3_{}.pdf'.format(num))
    plt.show()
def correlation_map(data,num):
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.savefig('图片/3_{}.pdf'.format(num))
    plt.show()
def count(data,attr,num):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8))
    sns.countplot(data[attr], label="数量")
    plt.ylabel('数量')
    plt.savefig('图片/3_{}.pdf'.format(num))
    plt.show()