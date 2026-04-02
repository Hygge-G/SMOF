# -*- coding: utf-8 -*-
"""
程序功能：论文复现
论文信息：
An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach, Part I: Solving Problems with Box Constraint
作者：(晓风)wangchao
最初建立时间：2019.03.26
最近修改时间：2019.04.01
最小化问题：DTLZ1,DTLZ2,DTLZ3
NSGA3的简单实现
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import uniformpoint,funfun,cal,GO,envselect,IGD,NDsort
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from pymoo.indicators.hv import Hypervolume
from pymoo.util.ref_dirs import get_reference_directions
random.seed(1)
np.random.seed(1)    # NumPy随机生成器




# # 生成参考方向（3目标）
# ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
# POP_SIZE = 100
# M = 3
# Z,N = uniformpoint(POP_SIZE,M)#生成一致性的参考解   参考点集合和个数
# print(Z)
# print(N)
# print(ref_dirs)
# print(M==ref_dirs)
# print(Z.shape)
# print(ref_dirs.shape)
# print(len(ref_dirs)==N)
# exit()


def get_model(normalized_X, y1, y2, y3):
    models = []
    # 决策变量1         K近邻，决策树，SVM
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y1, test_size=0.2, random_state=42)
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)

    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)

    model_SVR = SVR(kernel='rbf', C=100, epsilon=1)
    model_SVR.fit(X_train, y_train)

    models.append([model_KNN, model_DT, model_SVR])

    # 决策变量2      KNN，决策树，多层感知机
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y2, test_size=0.2, random_state=42)

    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)

    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)

    # 多层感知机
    model_MLP = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
    model_MLP.fit(X_train, y_train)

    models.append([model_KNN, model_DT, model_MLP])

    # 决策变量3      KNN，决策树，多层感知机
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y3, test_size=0.2, random_state=42)

    # KNN
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)

    # 决策树
    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)

    model_MLP = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
    model_MLP.fit(X_train, y_train)

    models.append([model_KNN, model_DT, model_MLP])
    return models
data = pd.read_csv(r'E:\User\手机定价\data\result_data\model_data_12000.csv')
data = data.drop(['耗时'], axis=1)
X = data.drop(['企业利润','平台利润','消费者总效益'], axis=1)
y = data[['企业利润','平台利润','消费者总效益']]
# 分离目标
y1 = y['企业利润']
y2 = y['平台利润']
y3 = y['消费者总效益']
y_total = {
    '企业利润':y1,
    '平台利润':y2,
    '消费者总效益':y3
}
scaler = MinMaxScaler()
normalized_X = scaler.fit_transform(X)
# 划分训练集和测试集
weight = [
    [0.1823, 0.6269, 0.1908],
    [0.0816, 0.6107, 0.3077],
    [0.0811, 0.7392, 0.1797]
]
models = get_model(normalized_X,y1,y2,y3)
def model_y(model_KNN,model_DT,model_RF,weight_y,X):
    # print(X)
    # X = np.array(X).reshape(1,-1)
    result = weight_y[0] * model_KNN.predict(X) + weight_y[1] * model_DT.predict(X) + weight_y[2] * model_RF.predict(X)
    return result


def func1(X):
    result = model_y(models[0][0],models[0][1],models[0][2],weight[0],X)
    return -result

def func2(X):
    result = model_y(models[1][0],models[1][1],models[1][2],weight[1],X)
    return -result

def func3(X):
    result = model_y(models[2][0],models[2][1],models[2][2],weight[2],X)
    return -result

func = [func1,func2,func3]

#参数设置
N_GENERATIONS = 200                                 # 迭代次数
POP_SIZE = 100                                      # 种群大小
D = 7                                # 测试函数选择，目前可供选择DTLZ1,DTLZ2,DTLZ3
M = 3                                               # 目标个数
t1 = 30                                             # 交叉参数t1
t2 = 20                                             # 变异参数t2
pc = 0.85                                              # 交叉概率
pm = 0.19                                             # 变异概率

###################################################################################################################################################################
#产生一致性的参考点和随机初始化种群

# 生成参考方向（3目标）
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
Z = ref_dirs
N = len(Z)
pop = funfun(POP_SIZE)#生成初始种群及其适应度值，真实的PF,自变量个数   N是参考点的个数
print('popsize',type(pop))
popfun = cal(pop,func)#计算适应度函数值

Zmin = np.array(np.min(popfun,0)).reshape(1,M)#求理想点

# 获得状态

random.seed(1)

#迭代过程
for i in range(N_GENERATIONS):
    print("第{name}次迭代".format(name=i+1))

    matingpool=random.sample(range(N),N)          # 仅仅是为了初始化操作 打乱
    # 更新pm和pc
    off = GO(pop[matingpool,:],t1,t2,pc,pm)#遗传算子,模拟二进制交叉和多项式变异
    offfun = cal(off,func)#计算适应度函数
    mixpop = copy.deepcopy(np.vstack((pop, off)))

    text = cal(mixpop, func)

    # 筛选
    temp = []
    for p in range(len(text)):
        if text[p][1] > 0:
            temp.append(p)
    mixpop = np.delete(mixpop, temp, axis=0)

    Zmin = np.array(np.min(np.vstack((Zmin,offfun)),0)).reshape(1,M)#更新理想点
    pop = envselect(mixpop,N,Z,Zmin,M,D,func)            # 更新       新的种群
    popfun = cal(pop,func)                               # 新的种群适应度

    # 以下




# 绘制PF
x = -popfun[:,0]
y = -popfun[:,1]
z = -popfun[:,2]


# score = IGD(popfun,PF)


print(len(x))
print(len(popfun))
# 创建图形
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')
# 添加网格线
ax.grid(True, color='grey', linestyle='-.', linewidth=0.3, alpha=0.2)
# 创建颜色映射
# my_cmap = plt.get_cmap('hsv')
# 添加散点图
sctt = ax.scatter(x, y, z, alpha=0.8, marker='*')      #  cmap=my_cmap,
# 添加颜色条
# fig.colorbar(sctt, ax=ax, shrink=0.3, aspect=5)
ax.set_title('NSGA-Ⅲ—base')

plt.show()

from pymoo.indicators.hv import Hypervolume
F = -popfun
# 假设解集的目标值矩阵为 F (n_samples, n_obj)
ref_point = np.array([12000,  8000 , 90000])# max(-popfun, axis=0) + 1e-3  # 动态设置（重要：需确保参考点足够大）
hv = Hypervolume(ref_point=ref_point)
hv_value = hv(F)
print('超体积：',hv_value)
print(popfun.shape)
print('size:',len(popfun))
from pymoo.indicators.igd import IGD
df = pd.read_csv(r'E:\User\手机定价\temp_data\true pareto.csv')
pf = np.array(df)
ind = IGD(pf)
print("IGD", ind(F))
len(F)


# 假设pareto_front是包含帕累托解的二维数组，每行表示一个解
pareto_front = F

# 转换为DataFrame并添加列名（根据目标函数命名）
df = pd.DataFrame(pareto_front, columns=['企业利润', '平台利润','消费者效益'])

# 保存到CSV（不保留索引，UTF-8编码）
df.to_csv('NSGA-Ⅲpareto_front.csv', index=False, encoding='utf-8-sig')

'''


超体积： 2726449482449.219
IGD 1694.7020389713764


超体积： 2799419755864.846
(496, 3)
IGD 712.9201980051646

超体积： 2875439255576.8574
(91, 3)
size: 91
IGD 1505.884302165712
组会
超体积： 2682690006667.951
(91, 3)
IGD 2102.099154195894
'''