import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from utils import uniformpoint,funfun,cal,GO,envselect
import copy
import random
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from pymoo.util.ref_dirs import get_reference_directions
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
random.seed(1)
np.random.seed(1)    # NumPy随机生成器



from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv(r'E:\User\手机定价\强化学习+NSGA3\output_只最前沿-修正.csv')



X_dcs = data.iloc[:, :-1]
y_dcs = data.iloc[:, -1]

X_train_dcs, X_test_dcs, y_train_dcs, y_test_dcs = train_test_split(
    X_dcs, y_dcs, test_size=0.3, random_state=42
)
scaler = MinMaxScaler()
X_train_dcs_scaled = scaler.fit_transform(X_train_dcs)
X_test_dcs_scaled = scaler.transform(X_test_dcs)

results = []

model_dcs = DecisionTreeClassifier()
model_dcs.fit(X_train_dcs_scaled, y_train_dcs)
print('x-test',X_test_dcs)
y_pred = model_dcs.predict(X_test_dcs_scaled)

def get_model(normalized_X, y1, y2, y3):
    models = []
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y1, test_size=0.2, random_state=42)
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)
    model_SVR = SVR(kernel='rbf', C=100, epsilon=1)
    model_SVR.fit(X_train, y_train)
    models.append([model_KNN, model_DT, model_SVR])
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y2, test_size=0.2, random_state=42)
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)
    model_MLP = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
    model_MLP.fit(X_train, y_train)
    models.append([model_KNN, model_DT, model_MLP])
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y3, test_size=0.2, random_state=42)
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
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
weight = [
    [0.1823, 0.6269, 0.1908],
    [0.0816, 0.6107, 0.3077],
    [0.0811, 0.7392, 0.1797]
]
models = get_model(normalized_X,y1,y2,y3)
def model_y(model_1,model_2,model_3,weight_y,X):
    result = weight_y[0] * model_1.predict(X) + weight_y[1] * model_2.predict(X) + weight_y[2] * model_3.predict(X)
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

#  定义 DQN 网络
class DQN(nn.Module):
    def __init__(self, state_dim=3, action_dim=11):
        super(DQN, self).__init__()
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# 经验回放（Replay Buffer）
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states))
    def __len__(self):
        return len(self.buffer)

def get_state(population):
    N = len(population)
    temp = -population
    temp = (temp - np.min(np.array(y), axis=0)) / (np.max(np.array(y), axis=0) - np.min(np.array(y), axis=0))  #
    Hqs = model_dcs.predict(temp)
    Hqs = sum(Hqs)/len(Hqs)

    if N == 0:
        return (0.0, 0.0, 0.0)
    population = (population - np.min(-np.array(y), axis=0)) / (np.max(-np.array(y), axis=0) - np.min(-np.array(y), axis=0))
    con_sum = 0.0
    max_objs = [-float('inf')] * 3
    min_objs = [float('inf')] * 3

    for ind in population:
        con_sum += sum(ind)
        for j in range(3):
            val = ind[j]
            if val > max_objs[j]:
                max_objs[j] = val
            if val < min_objs[j]:
                min_objs[j] = val

    con = con_sum / N

    range_sum = 0.0
    for j in range(3):
        delta = max_objs[j] - min_objs[j]
        if delta <= 1e-6:
            delta = 1e-6
        range_sum += delta

    div = 1.0 / range_sum if range_sum != 0 else float('inf')
    print(con,div,1-Hqs)
    return [con,div,1-Hqs]

# 4. DQN
def train_dqn(model_name):
    num_episodes = 200
    batch_size = 50
    gamma = 0.5
    lr = 0.01

    epsilon_start = 1.0
    epsilon_end = 0.2
    epsilon_decay = 1000

    target_update_interval = 50
    replay_buffer_capacity = 1000

    N_GENERATIONS = 200
    POP_SIZE = 100
    M = 3
    D = 7
    t1 = 30
    t2 = 20
    # 创建网络
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    all_rewards = []
    Cr_values = np.round(np.linspace(0.4, 0.9, 11), 4)  # 注意num参数改为11
    F_values = np.round(np.linspace(0.01, 0.21, 11), 4)
    actions = [
        [float(Cr_values[i]), float(F_values[i])]
        for i in range(11)
    ]
    # 训练过程
    for episode in range(num_episodes):
        print("第{name}次优化".format(name=episode + 1))
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        Z = ref_dirs
        N = len(Z)
        pop = funfun(POP_SIZE)                                   # 生成初始种群及其适应度值
        popfun = cal(pop,func)
        Zmin = np.array(np.min(popfun, 0)).reshape(1, M)         # 初始理想点
        for gen in range(N_GENERATIONS):
            print("第{name}次迭代".format(name=gen + 1))
            popfun = cal(pop, func)                              # 计算适应度
            state = get_state(popfun)                            # 3个参数
            state_tensor = torch.FloatTensor(state).unsqueeze(0)        # 调整成1*3
            episode_reward = 0
            # 计算当前 epsilon   更新
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                np.exp(-1. * episode / epsilon_decay)
            # 根据 epsilon 贪心选择动作
            # if random.random() < epsilon:
            action = random.randint(0,len(actions)-1)
            # else:
            #     with torch.no_grad():
            #         q_values = policy_net(state)
            #         action = q_values.argmax(dim=1).item()
            pc,pm = actions[action]
            matingpool = random.sample(range(len(pop)), len(pop))
            off = GO(pop[matingpool, :], t1, t2, pc, pm)
            offfun = cal(off, func)
            mixpop = copy.deepcopy(np.vstack((pop, off)))

            text = cal(mixpop, func)
            temp = []
            for p in range(len(text)):
                if text[p][1] > 0:
                    temp.append(p)
            mixpop = np.delete(mixpop, temp, axis=0)

            Zmin = np.array(np.min(np.vstack((Zmin, offfun)), 0)).reshape(1, M)
            # 与环境进行一步交互
            next_pop = envselect(mixpop, N, Z, Zmin, M, D, func)
            next_popfun = cal(next_pop, func)
            next_state = get_state(next_popfun)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            reward = sum(state[0:-1]) - sum(next_state[0:-1])  # 奖励设计                       computer_reward(state, next_state)  # 两种状态计算奖励

            pop = np.copy(next_pop)
            # 将 transition 存到经验回放中
            replay_buffer.push(
                state_tensor.squeeze(0).numpy(),
                action,
                reward,
                next_state_tensor.squeeze(0).numpy()
            )
            episode_reward += reward

            # 每步都尝试训练（如果缓冲区够大）
            if len(replay_buffer) >= batch_size:
                # 从回放缓冲区采样
                states_b, actions_b, rewards_b, next_states_b = replay_buffer.sample(batch_size)
                states_b = torch.FloatTensor(states_b)
                actions_b = torch.LongTensor(actions_b)
                rewards_b = torch.FloatTensor(rewards_b)
                next_states_b = torch.FloatTensor(next_states_b)

                # 计算 Q(s, a)
                # q_values = policy_net(torch.FloatTensor(np.hstack([states_b, actions_b])))         # 得到每个状态的Q值，即奖励
                q_values = policy_net(states_b)
                # print(q_values)

                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)


                # 计算 Q'(s', a') 来 更新目标
                with torch.no_grad():
                    # 使用target_net来计算 max Q'(s', a')
                    next_q_values = target_net(next_states_b)           # 下一阶段的q值，即奖励
                    max_next_q_values = next_q_values.max(dim=1)[0]
                    # 如果结束，那么目标是 reward；否则是 reward + gamma * max Q'(s', a')
                    target_q_values = rewards_b + gamma * (1 - gen//200) * max_next_q_values

                    # target_q_values = q_values + α(rewards + gamma * max_next_q_values - q_values)
                # 计算损失
                loss = nn.MSELoss()(q_values, target_q_values)

                # 反向传播和更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_rewards.append(episode_reward)

        # 每隔一段时间更新目标网络
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 打印训练信息
        print(f"Episode {episode+1}, Epsilon: {epsilon:.3f}, Reward: {episode_reward}")
        if (episode+1)%5==0:
            torch.save(policy_net.state_dict(), 'dqn_policy-Hqs-test-0.5gamm-reward=2'+str(episode+1)+'.pth')
            print(f"model saved:",'dqn_policy-'+str(episode +1)+'.pth')

    # 保存训练好的网络
    torch.save(policy_net.state_dict(), model_name)
    print(f"model saved: {model_name}")

    return all_rewards

import copy
def test_dqn(trained_model_path):
    """
    使用训练好的DQN模型在迷宫环境中测试 num_episodes 次，
    参数:
        trained_model_path: str, 已保存的模型文件路径，例如 'dqn_policy.pth'
    """
    N_GENERATIONS = 200
    POP_SIZE = 100
    t1 = 30
    t2 = 20
    M = 3                   # 目标函数个数
    D = 7                   # 决策变量个数
    # 2. 构建与训练时相同的网络结构，并加载训练好的模型参数
    policy_net = DQN(state_dim=3, action_dim=11)
    policy_net.load_state_dict(torch.load(trained_model_path))
    policy_net.eval()  # 推断模式

    # 生成action集合
    Cr_values = np.round(np.linspace(0.4, 0.9, 11), 4)  # 注意num参数改为11
    F_values = np.round(np.linspace(0.01, 0.21, 11), 4)
    # 创建参数对组合（按索引位置一一对应）
    actions = [
        [float(Cr_values[i]), float(F_values[i])]
        for i in range(11)
    ]
    pop = funfun(POP_SIZE)
    Z, N = uniformpoint(POP_SIZE, M)
    popfun = cal(pop,func)
    Zmin = np.array(np.min(popfun, 0)).reshape(1, M)      # 理想点
    for i in range(N_GENERATIONS):
        print("第{name}次迭代".format(name=i + 1))
        popfun = cal(pop,func)
        state = get_state(popfun)
        state = torch.FloatTensor(state).unsqueeze(0)

        # 动作选择    随机选择动作
        # if random.random()<0.5:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.argmax(dim=1).item()
        #else:
        # action = random.randint(0,len(actions)-1)
        print(action)
        pc,pm = actions[action] #   actions[action]
        matingpool = random.sample(range(len(pop)), len(pop))                      # 仅仅是为了初始化操作 打乱
        off = GO(pop[matingpool, :], t1, t2, pc, pm)                               # 遗传算子,模拟二进制交叉和多项式变异
        offfun = cal(off, func)                                                    # 计算适应度函数   新的子代种群的适应度
        mixpop = copy.deepcopy(np.vstack((pop, off)))                              # 整合新的种群，即为需要排序的种群

        text = cal(mixpop, func)
        temp = []
        for p in range(len(text)):
            if text[p][1] > 0:
                temp.append(p)
        mixpop = np.delete(mixpop, temp, axis=0)

        Zmin = np.array(np.min(np.vstack((Zmin, offfun)), 0)).reshape(1, M)        # 更新理想点
        new_pop = envselect(mixpop, N, Z, Zmin, M, D, func)                        # 更新       新的种群   通过参考点排序的方式，精英保留新的种群
        pop = np.copy(new_pop)                                                     # 更新种群
    return pop

if __name__ == "__main__":
    # random.seed(1)     # 42 是81   1 是89
    # model_name = 'dqn_policy-Hqs-test-gam-0.5-.pth'
    # rewards = train_dqn(model_name)
    save_HV = []
    save_IGD = []
    for i in range(20,141,5):
        random.seed(1)
        np.random.seed(1)
        moedel_x = i
        model_name = 'dqn_policy-Hqs-test-0.5gamm-reward=2'+str(moedel_x)+'.pth'
        pop = test_dqn(model_name)
        popfun = cal(pop, func)
        # x = -popfun[:, 0]
        # y = -popfun[:, 1]
        # z = -popfun[:, 2]
        # import matplotlib.pyplot as plt
        # # 创建图形
        # fig = plt.figure(figsize=(15, 9))
        # ax = fig.add_subplot(111, projection='3d')
        # sctt = ax.scatter(x, y, z, alpha=0.8, marker='*')  # cmap=my_cmap,
        # ax.set_title('NSGA-RL')
        # plt.show()

        F= -popfun
        ref_point = np.array([12000, 8000, 80000])  # max(-popfun, axis=0) + 1e-3  # 动态设置（重要：需确保参考点足够大）
        hv = Hypervolume(ref_point=ref_point)
        hv_value = hv(F)
        print('超体积：', hv_value)
        print(popfun.shape)
        print('size:', len(popfun))
        from pymoo.indicators.igd import IGD

        df = pd.read_csv(r'E:\User\手机定价\temp_data\true pareto.csv')
        pf = np.array(df)
        ind = IGD(pf)
        print("IGD", ind(F))
        save_HV.append(hv_value)
        save_IGD.append(ind(F))
    for i in range(len(save_HV)):
        print('第{name}组：'.format(name=i*5+20))
        print('超体积：', save_HV[i])
        print('IGD值：', save_IGD[i])
'''

'''

