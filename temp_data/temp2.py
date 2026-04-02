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
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler
# 导入机器学习模型
from sko.GA import GA
data = pd.read_csv(r'E:\User\手机定价\data\result_data\model_data_12000.csv')  # test_model_data测试6.csv
data.head()

data = data.drop(['耗时'], axis=1)
# 划分特征和目标
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
#　归一化
scaler = MinMaxScaler()
normalized_X = scaler.fit_transform(X)
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(normalized_X, y3, test_size=0.2, random_state=42)


'''
设计目标函数，三个模型预测结果的加权，最后得到的值与真实值的差值，也就是RC值，还是知识R²值。 
三个模型，单个目标，只优化权重    KNN，决策树，多层感知机       针对目标2
'''
count = 0
r2_2 = []
rmse_2 = []

X_train, X_test, y_train, y_test = train_test_split(normalized_X, y2, test_size=0.2, random_state=42)
# k近邻
model_1 = KNeighborsRegressor()
model_1.fit(X_train, y_train)
# 决策树
model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X_train, y_train)
# 多层感知机、
model_3 = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
model_3.fit(X_train, y_train)

def fun(a, b, c,model_1,model_2,model_3):

    y_pred_1 = model_1.predict(X_test)

    y_pred_2 = model_2.predict(X_test)


    y_pred_3 = model_3.predict(X_test)

    y_pred = a * y_pred_1 + b * y_pred_2 + c * y_pred_3

    mse = mean_squared_error(y_test, y_pred)
    print(f'R Mean Squared Error: {np.sqrt(mse)}')
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmse_2.append(rmse)
    return r2


def obj_fun(X):
    a, b, c = X
    a, b, c = a / sum(X), b / sum(X), c / sum(X)
    global model_1, model_2, model_3
    R2 = fun(a, b, c,model_1, model_2, model_3)
    r2_2.append(R2)
    global count
    print(count + 1, R2)
    count += 1
    # 适应度函数，应该是约接近1越好，而不是简单的越大或者越小
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
print('最优值 f(x):', best_y)


temp_np = np.array([r2_2])
avera_r2 = temp_np.reshape(100,50).mean(axis=1)
print(avera_r2.shape)  # 输出 (300,)
temp_np = np.array([rmse_2])
avera_rmse = temp_np.reshape(100,50).mean(axis=1)
print(avera_rmse.shape)  # 输出 (300,)

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
import numpy as np

x3_ave = range(1,len(avera_rmse)+1)
x3 = range(1,len(r2_2)+1)

y3_r2 = avera_r2#averages
y3_rmse = avera_rmse

r2 = r2_2
rmse = rmse_2
# 创建2x2子图布局
fig, axes = plt.subplots(2, 2, figsize=(12, 8),dpi=1000)

# 绘制曲线（自定义颜色、线宽、标签）

axes[0, 0].plot(x3_ave, y3_r2, color='orange', linestyle='-',linewidth=1) # label='决定系数均值',
axes[0, 0].set_title('R2平均迭代值过程', fontsize=10)
axes[0, 0].set_xlabel('迭代次数', fontsize=12)
axes[0, 0].set_ylabel('R2值', fontsize=12)
# axes[0, 0].legend(loc='upper right', fontsize=10)
axes[0, 0].grid(True, linestyle='-', alpha=0.4)

# 第二章子图
axes[0, 1].plot(x3_ave, y3_rmse, color='blue', linestyle='-',linewidth=1) # label='决定系数均值',
axes[0, 1].set_title('Rmse平均迭代值过程', fontsize=10)
axes[0, 1].set_xlabel('迭代次数', fontsize=12)
axes[0, 1].set_ylabel('Rmse值', fontsize=12)
axes[0, 1].grid(True, linestyle='-', alpha=0.4)

# 第三张子图
axes[1, 0].plot(x3, r2, color='green', linestyle='-',linewidth=1) # label='决定系数均值',
axes[1, 0].set_title('R2迭代值过程', fontsize=10)
axes[1, 0].set_xlabel('迭代次数', fontsize=12)
axes[1, 0].set_ylabel('R2值', fontsize=12)
axes[1, 0].grid(True, linestyle='-', alpha=0.4)

# 第四张子图
axes[1, 1].plot(x3, rmse, color='purple', linestyle='-',linewidth=1) # label='决定系数均值',
axes[1, 1].set_title('RMSE迭代值过程', fontsize=10)
axes[1, 1].set_xlabel('迭代次数', fontsize=12)
axes[1, 1].set_ylabel('RMSE值', fontsize=12)
axes[1, 1].grid(True, linestyle='-', alpha=0.4)
# 保存图像（需在plt.show()之前）
plt.tight_layout()
plt.savefig('y2图片四合一.png', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()