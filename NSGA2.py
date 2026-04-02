from function import *
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
random.seed(42)
np.random.seed(42)    # NumPy随机生成器
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

def func1(X):
    result = model_y(models[0][0],models[0][1],models[0][2],weight[0],X)
    return -result

def func2(X):
    result = model_y(models[1][0],models[1][1],models[1][2],weight[1],X)
    return -result

def func3(X):
    result = model_y(models[2][0],models[2][1],models[2][2],weight[2],X)
    return -result


def crossover(x, y):
    r = random.random()
    # print('x,y:',x,y)
    if r > 0.5:
        return mutation([(per_x + per_y)/2 for per_x, per_y in zip(x, y)])
    else:
        return mutation([(per_x - per_y)/2 for per_x, per_y in zip(x, y)])


def mutation(solution):
    min_v = 0
    max_v = 1
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = [min_x + (max_x - min_x) * random.random() for j in range(0,x_size)]            # 生成新的个体
    return solution


# NSGA2主循环
def main_loop(pop_size, max_gen, init_population):
    gen_no = 0
    population_P = init_population.copy()
    while gen_no < max_gen:
        population_R = population_P.copy()
        # 根据P(t)生成Q(t),R(t)=P(t)vQ(t)
        while len(population_R) != 2 * pop_size:  # 增加，两个交换—> 三个交换      不对，一组解里面，包含3个值不就可以了
            x = random.randint(0, pop_size - 1)
            y = random.randint(0, pop_size - 1)
            population_R.append(crossover(population_P[x], population_P[y]))
        # 对R(t)计算非支配前沿
        objective1 = [func1(population_R[i]) for i in range(2 * pop_size)]
        objective2 = [func2(population_R[i]) for i in range(2 * pop_size)]
        objective3 = [func3(population_R[i]) for i in range(2 * pop_size)]
        fronts = fast_non_dominated_sort(objective1, objective2, objective3)
        # print('fronts:', len(fronts))
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
            level_distance = crowed_distance_assignment(objective1, objective2, objective3, fronts[level])
            sort_solution = sorted(fronts[level], key=lambda x: level_distance[fronts[level].index(x)], reverse=True)
            for i in range(pop_size - len(population_P_next)):
                choose_solution.append(sort_solution[i])
                population_P_next.append(population_R[sort_solution[i]])
        # 得到P(t+1)重复上述过程
        population_P = population_P_next.copy()
        if gen_no % 50 == 0:
            best_obj1 = [func1(population_P[i]) for i in range(pop_size)]
            best_obj2 = [func2(population_P[i]) for i in range(pop_size)]
            best_obj3 = [func3(population_P[i]) for i in range(pop_size)]
            f = fast_non_dominated_sort(best_obj1, best_obj2, best_obj3)
            print(f'generation {gen_no}, first front:')
            for s in f[0]:                                    # 打印最前沿的解
                print(population_P[s], end=' ')
            print('\n')
        gen_no += 1
    return best_obj1, best_obj2, best_obj3

#data = pd.read_csv(r'E:\User\手机定价\data\result_data\model_data.csv')
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
    X = np.array(X).reshape(1,-1)
    result = weight_y[0] * model_KNN.predict(X) + weight_y[1] * model_DT.predict(X) + weight_y[2] * model_RF.predict(X)
    return result

x_size = 7
min_x = 0
max_x = 1
pop_size = 100
max_gen = 200                    # min_v + (max_v - min_v) * random.random()
population_P = [[min_x + (max_x - min_x) * random.random() for j in range(0,x_size)] for i in range(0, pop_size)]
v1, v2, v3 = main_loop(pop_size, max_gen, population_P)

x = list(map(lambda x: -x, v1))   # -v1,
y = list(map(lambda x: -x, v2))   # -v2
z = list(map(lambda x: -x, v3))   # -v3

# 创建图形
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')
# 添加网格线
ax.grid(True, color='grey', linestyle='-.', linewidth=0.3, alpha=0.2)
# 创建颜色映射
# my_cmap = plt.get_cmap('hsv')
# 添加散点图
sctt = ax.scatter(x, y, z, alpha=0.8, marker='*')      #  cmap=my_cmap,
# 添加颜色条
# fig.colorbar(sctt, ax=ax, shrink=0.3, aspect=5)
ax.set_title('NSGA-2')
# plt.savefig('NSGA-2帕累托前沿面-没有颜色条-最大化-test.png', dpi=1000)
# 显示图形
plt.show()

from pymoo.indicators.hv import Hypervolume
F = -np.column_stack((v1, v2, v3))
# 假设解集的目标值矩阵为 F (n_samples, n_obj)
ref_point = np.array([12000,  8000 , 90000])# max(-popfun, axis=0) + 1e-3  # 动态设置（重要：需确保参考点足够大）
hv = Hypervolume(ref_point=ref_point)
hv_value = hv(F)
print('超体积：',hv_value)
from pymoo.indicators.igd import IGD
df = pd.read_csv(r'E:\User\手机定价\temp_data\true pareto.csv')
pf = np.array(df)
ind = IGD(pf)
print("IGD", ind(F))

