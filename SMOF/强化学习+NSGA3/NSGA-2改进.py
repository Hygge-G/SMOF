from function import *
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
random.seed(1)
np.random.seed(1)

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
# 快速非支配排序
def fast_non_dominated_sort(values1, values2, values3):
    size = len(values1)
    dominate_set = [[] for _ in range(size)]  # 解p支配的解集合
    dominated_count = [0 for _ in range(size)]  # 支配p的解数量
    solution_rank = [0 for _ in range(size)]  # 每个解的等级
    fronts = [[]]

    for p in range(size):
        dominate_set[p] = []
        dominated_count[p] = 0
        for q in range(size):
            if values1[p] <= values1[q] and values2[p] <= values2[q] and values3[p] <= values3[q] \
                    and ((values1[p] == values1[q]) + (values2[p] == values2[q]) + (
                    values3[p] == values3[q])) != 3:  # 确保不都等于
                dominate_set[p].append(q)
            elif values1[q] <= values1[p] and values2[q] <= values2[p] and values3[q] <= values3[p] \
                    and ((values1[q] == values1[p]) + (values2[q] == values2[p]) + (values3[q] == values3[p])) != 3:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            solution_rank[p] = 0
            fronts[0].append(p)

    level = 0
    while fronts[level]:
        Q = []
        for p in fronts[level]:
            for q in dominate_set[p]:  # 支配的解集合
                dominated_count[q] -= 1  # 支配的解数量-1
                if dominated_count[q] == 0:  # 就是没了
                    solution_rank[q] = level + 1  # 确定解的等级
                    if q not in Q:
                        Q.append(q)
        level = level + 1
        fronts.append(Q)
    del fronts[-1]
    return fronts

# 计算拥挤距离
def crowed_distance_assignment(values1, values2, values3, front):
    length = len(front)
    sorted_front1 = sorted(front, key=lambda x: values1[x])
    sorted_front2 = sorted(front, key=lambda x: values2[x])
    sorted_front3 = sorted(front, key=lambda x: values3[x])

    dis_table = {sorted_front1[0]: np.inf, sorted_front1[-1]: np.inf,
                 sorted_front2[0]: np.inf, sorted_front2[-1]: np.inf,
                 sorted_front3[0]: np.inf, sorted_front3[-1]: np.inf}
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table.get(k, 0) + (values1[sorted_front1[i + 1]] - values1[sorted_front1[i - 1]]) / (
                    max(values1) - min(values1))
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table[k] + (values2[sorted_front2[i + 1]] - values2[sorted_front2[i - 1]]) / (
                    max(values2) - min(values2))
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table[k] + (values3[sorted_front3[i + 1]] - values3[sorted_front3[i - 1]]) / (
                    max(values3) - min(values3))
    distance = [dis_table[a] for a in front]
    return distance


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


import copy
from utils import funfun,cal,GO,NDsort,lastselection

# NSGA2主循环
def main_loop(pop_size, max_gen,t1, t2, pc, pm):
    gen_no = 0
    pop = funfun(pop_size)                                               #生成初始种群及其适应度值，真实的PF,自变量个数   N是参考点的个数
    while gen_no < max_gen:
        print("第{name}次迭代".format(name=gen_no + 1))
        matingpool = random.sample(range(pop_size), pop_size)            # 仅仅是为了初始化操作 打乱
        off = GO(pop[matingpool, :], t1, t2, pc, pm)  # 遗传算子,模拟二进制交叉和多项式变异  获得新的种群

        text = cal(off, func)
        # 筛选
        temp = []
        for p in range(len(text)):
            if text[p][1] > 0:
                temp.append(p)
        off = np.delete(off, temp, axis=0)
        print(len(off))
        mixpop = copy.deepcopy(np.vstack((pop, off)))
        popfun = cal(mixpop, func)  # 计算适应度函数

        population_R = mixpop.copy()

        fronts = fast_non_dominated_sort(popfun[:,0], popfun[:,1], popfun[:,2])
        # 获取P(t+1)，先从等级高的fronts复制，然后在同一层front根据拥挤距离选择
        population_P_next = []
        choose_solution = []
        level = 0
        while len(population_P_next) + len(fronts[level]) <= pop_size:  # 前面能放下的，符合条件的，先放进去
            for s in fronts[level]:
                choose_solution.append(s)  # 已经选择的
                population_P_next.append(population_R[s])
            level += 1
        if len(population_P_next) != pop_size:  # 选择完成之后，还没有满，剩下的选择按照拥挤度排序
            level_distance = crowed_distance_assignment(popfun[:,0], popfun[:,1], popfun[:,2], fronts[level])
            sort_solution = sorted(fronts[level], key=lambda x: level_distance[fronts[level].index(x)], reverse=True)
            for i in range(pop_size - len(population_P_next)):
                choose_solution.append(sort_solution[i])
                population_P_next.append(population_R[sort_solution[i]])

        pop = population_P_next.copy()
        pop = np.array(pop)
        gen_no += 1
    return pop


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
    # print(X.shape)
    # X = np.array(X).reshape(1,-1)
    result = weight_y[0] * model_KNN.predict(X) + weight_y[1] * model_DT.predict(X) + weight_y[2] * model_RF.predict(X)
    return result

x_size = 7
min_x = 0
max_x = 1
pop_size = 100
max_gen = 200
t1 = 15                                             # 交叉参数t1
t2 = 20                                             # 变异参数t2
pc = 0.9                                              # 交叉概率
pm = 1/3                                              # 变异概率
# population_P = funfun(pop_size)
pop = main_loop(pop_size, max_gen,t1, t2, pc, pm)

popfun = cal(pop,func)
x = -popfun[:,0]
y = -popfun[:,1]
z = -popfun[:,2]

# 创建图形
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')

# 创建颜色映射
# my_cmap = plt.get_cmap('hsv')
# 添加散点图
sctt = ax.scatter(x, y, z, alpha=0.8, marker='*')      #  cmap=my_cmap,

ax.set_title('NSGA-Ⅱ-SBX')

# 显示图形
plt.show()

from pymoo.indicators.hv import Hypervolume


F = -popfun
ref_point = np.array([12000,  8000 , 90000])
hv = Hypervolume(ref_point=ref_point)
hv_value = hv(F)
print('超体积：',hv_value)
from pymoo.indicators.igd import IGD
df = pd.read_csv(r'E:\User\手机定价\temp_data\true pareto 去除小于零.csv')
pf = np.array(df)
ind = IGD(pf)
print("IGD", ind(F))

len(F)


# 假设pareto_front是包含帕累托解的二维数组，每行表示一个解
# pareto_front = F
#
# # 转换为DataFrame并添加列名（根据目标函数命名）
# df = pd.DataFrame(pareto_front, columns=['企业利润', '平台利润','消费者效益'])
#
# # 保存到CSV（不保留索引，UTF-8编码）
# df.to_csv('NSGA-Ⅱpareto_front.csv', index=False, encoding='utf-8-sig')



'''
1915932445970.5344
1811698171415.1785
1811698171415.1785

超体积： 3724069643578.0967
IGD 2524.9702304970083

超体积： 4431743522675.557
IGD 1891.9439867648089

超体积： 3726415395662.8916     100，200 0。7，0.1
IGD 2528.2760919187563

超体积： 3726415395662.8916
IGD 2528.2760919187563
'''