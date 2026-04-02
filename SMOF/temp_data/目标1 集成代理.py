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
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv(r'E:\User\手机定价\data\result_data\model_data_12000.csv')  # test_model_data测试6.csv
print(data.head())
data = data.drop(['耗时'], axis=1)
# 划分特征和目标
X = data.drop(['企业利润','平台利润','消费者总效益'], axis=1)
y = data[['企业利润','平台利润','消费者总效益']]
scaler = MinMaxScaler()
normalized_X = scaler.fit_transform(X)
# 分离目标
y1 = y['企业利润']
y2 = y['平台利润']
y3 = y['消费者总效益']


# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from sko.GA import GA

'''      
设计目标函数，三个模型预测结果的加权，最后得到的值与真实值的差值，也就是RC值，还是知识R²值。优化第一个，三个模型为：K近邻，决策树，SVM
三个模型，单个目标，只优化权重         # 优化第一个函数的权重
'''
count = 0
temp = []


def fun(a, b, c):
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y1, test_size=0.2, random_state=42)
    # k近邻
    model_1 = KNeighborsRegressor()
    model_1.fit(X_train, y_train)
    y_pred_1 = model_1.predict(X_test)

    # 决策树
    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(X_train, y_train)
    # 预测
    y_pred_2 = model_2.predict(X_test)
    # 计算指标
    # r2_2 = r2_score(y_test, y_pred_2)

    # SVR 支持向量机回归
    model_3 = SVR(kernel='rbf', C=100, epsilon=1)
    model_3.fit(X_train, y_train)
    y_pred_3 = model_3.predict(X_test)

    y_pred = a * y_pred_1 + b * y_pred_2 + c * y_pred_3
    #     mse = mean_squared_error(y_test, y_pred)
    #     print(f'R Mean Squared Error: {np.sqrt(mse)}')
    r2 = r2_score(y_test, y_pred)
    return r2


def obj_fun(X):
    a, b, c = X
    a, b, c = a / sum(X), b / sum(X), c / sum(X)
    R2 = fun(a, b, c)
    temp.append(R2)
    global count
    print(count + 1, R2)
    count += 1
    return 1 - R2


# 设置遗传算法参数
ga = GA(func=obj_fun,
        n_dim=3,  # 变量维度
        size_pop=50,  # 种群大小
        max_iter=100,  # 最大迭代次数
        lb=[0] * 3,  # 变量下界
        ub=[1] * 3,  # 变量上界
        precision=1e-7,
        prob_mut=0.01)  # 精度

# 运行算法
best_x, best_y = ga.run()

# 输出结果
print('最优解 x:', best_x)
print('最优值 f(x):', 1 - best_y)