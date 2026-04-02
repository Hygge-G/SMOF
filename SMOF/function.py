'''
定义各种需要的函数
'''
import random
import subprocess
import sys

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

# subprocess.check_call([sys.executable, "-m", "pip", "install", "snownlp"])

# 企业定义
class company:
    def __init__(self,name,product1,product2,product3): # 两个产品
        self.name = name                  # 企业代号
        self.product1 = product1          # 企业中产品1
        self.product2 = product2          # 企业中产品2
        self.product3 = product3          # 企业中产品3

# 产品定义
class product:
    def __init__(self,price,trade_price,cost_price,tech,G,recommend,review,chanl):
        self.price = price                # 价格，即售价
        self.trade_price = trade_price    # 批发价
        self.cost_price = cost_price      # 成本价
        self.tech = tech                  # 科技含量
        self.G = G                        # 代数
        self.recommend = recommend        # 推荐指数           # 不是最新款推荐指数就是0
        self.sell_num = 0                 # sell_num          # 销售量
        self.review = review            # 产品评论集，是一个集合
        self.chanl = chanl                # 销售渠道

# 评论定义
class comment:                      # 评论集直接存储到另一个文本上
    def __int__(self,id,time,emo):
        self.id = id               #评论人
        self.time = time           #评论时间  这个命名注意和时间函数冲突
        self.emo = emo             #评论情感倾向

# 渠道定义
class channel:
    def __init__(self,productA1,productA2,productA3,productB1,productB2,productB3):     # 渠道是否应该包括手机，一个手机的渠道？渠道所对应的产品，渠道可以出售所有的产品
        self.productA1 = productA1                  # 产品
        self.productA2 = productA2                  # 销售渠道
        self.productA3 = productA3                  # 销售渠道
        self.productB1 = productB1                  # 销售渠道
        self.productB2 = productB2                  # 销售渠道
        self.productB3 = productB3                  # 销售渠道


# 消费者定义
class customer:
    def __init__(self,id,sen_price,sen_tech,benefit):
        self.flag = 0                    # 表示没有买新机，购买是单次行为
        self.id = id                     # 用户id
        self.sen_price = sen_price       # 价格敏感度
        self.sen_tech = sen_tech         # 技术敏感度  np.random.dirichlet(np.ones(4))              #屏幕，电池，系统，处理器
        self.benefit = benefit           # 消费者效益

    def STEP1(self,platform,person):                # 购买行为综合考量价格敏感度、技术敏感度、手机折旧度，渠道，推广效应，产品
        cur_phone = []
        flt_phone = []
        for chan in platform:                                     # 访问渠道
            for shouji in chan:                         # 遍历该渠道所有属性
                key = computer_fit(shouji,person.sen_price,person.sen_tech,person.benefit)        # 计算效益
                if key:
                    flt_phone.append(key)
                    cur_phone.append(shouji)    # 加入预选集合
        return flt_phone,cur_phone

    def STEP2(self,flt_phone,cur_phone,platform,pis):                # 广告推荐，群体效应对决策的影响，对于每一个产品而言
        cur_fit = []
        for i in range(len(cur_phone)):
            ifu_in = computer_inifu(cur_phone[i],pis)    # 外部效应，销量得改一下
            # ifu_out = computer_outifu(cur_phone[i],get_ano_phone(platform,cur_phone[i]))          # 内部效应，评论
            ifu_people = (1+ifu_in-0.5)#*(1+ifu_out)
            flt_phone[i] *= ifu_people * (1+cur_phone[i].recommend)     # 这里一代产品的recommend为0
            cur_fit.append(flt_phone[i])                # 更新后的产品效益

        result_fit = max(cur_fit)
        result = cur_phone[cur_fit.index(result_fit)]
        return result_fit,result            #返回最后购买的产品,和相对应的效益

    def buy(self,person,product):               # 购买行为，需要什么：渠道，产品
        person.flag = 1              # 标记已经购买
        product.sell_num += 1               # 产品销量+1


def computer_inifu(phone,pis):         # 评论操作是一个单独的模块。
    comment = phone.review
    # 根据分布得到停止位置
    # 读取对应位置的综合评价
    # print(phone.review)
    # print(pis)
    emo = random.choices(comment,list(pis),k=1)
    if isinstance(emo[0],str):
        # print(len(comment),len(pis))
        # print(emo[0].strip(),len(emo))
        # print(type(emo[0].strip()))
        return float(emo[0].strip())
    else:
        return emo[0]

def computer_outifu(phone1,phone2):

    n1 = phone1.sell_num
    n2 = phone2.sell_num
    return (n1)/(n1+n2+0.0000001)      # 这里有问题，得想想怎么平滑处理   销量并不在一个量级啊·

def get_ano_phone(platform,cur_phone):
    for chan in range(len(platform)):
        for phone in range(len(platform[chan])):
            if platform[chan][phone] == cur_phone:
                return platform[not chan][phone]
    print("找不到对应渠道的产品，产品有改动")
    return None

def computer_fit(shouji,sen_price,sen_tech,benefit):       #  新品效益与当前产品效益的比较
    sum_tech = 0
    for i in range(len(sen_tech)):
        sum_tech += shouji.tech[i] * sen_tech[i]
    fit = sum_tech-shouji.price*sen_price
    if fit>benefit:
        return fit
    else:
        return False

def get_review_path(path):
    data = pd.read_csv(path)
    temp = data['输出']
    return list(temp)

def get_pi(n):
    data = np.random.rand(200)
    data = data/sum(data)
    pi = []
    temp = 1
    for idx in range(len(data)):
        for j in range(idx):
            temp = temp*(1-data[j])
        pi.append(temp)
    return pi


def Print(person,product,customer_prifit,day):
    print("用户{0}在第{4}天从渠道{1}购买了公司的{2}代产品，效益为：{3}".format(person.id,product.chanl,product.G,round(customer_prifit,3),day))


def Print_cumsomers(customers):
    for per in customers:
        print("第{0}个用户".format(per.id))
        print('技术敏感度：',per.sen_tech)
        print('价格敏感度：', per.sen_price)
        print('手机损耗度：', per.phone_loss)


def A_prifit(platform):                  # 企业A的总利润     成本如何计算的问题
    sumA = 0
    # 渠道1
    for mi in platform[0]:
        sumA += mi.sell_num * (mi.price - mi.cost_price)       # 售价 - 成本价 - 推荐成本  不是第三代的，推荐成本就是0
        sumA -= mi.sell_num * mi.price * 0.036                     # 减去佣金率和服务费用
    # 渠道2
    for mi in platform[1]:
        sumA += mi.sell_num * (mi.trade_price - mi.cost_price)       # 批发价-成本价
    sumA -= ((mi.recommend)*(mi.recommend)*1000)/2                   # 推荐成本
    return round(sumA,2)


def platform_prifit(platform):           # 平台总利润
    sumC = 0
    for per_product in platform[0]:
        sumC += per_product.sell_num * per_product.price * 0.036
    for per_product in platform[1]:
        sumC += per_product.sell_num * (per_product.price - per_product.trade_price)
    return round(sumC,2)

def sum_benefit(customers):              # 消费者总效益
    sum_fit = 0
    for person in customers:
        if person.flag==1:
            sum_fit += person.benefit
    return round(sum_fit,2)


def print_num(customers):           # 查看销售量
    x = 0
    for i in customers:
        x += i.flag
    return x


def get_model(normalized_X,y1,y2,y3):
    models = []
    # 决策变量1         K近邻，决策树，SVM
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y1, test_size=0.2, random_state=42)
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
    # y_pred = model_KNN.predict(X_test)
    # # r2 = r2_score(y_test, y_pred)
    # # print(r2)

    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)
    # y_pred = model_DT.predict(X_test)
    # # r2 = r2_score(y_test, y_pred)
    # # print(r2)

    model_SVR = SVR(kernel='rbf', C=100, epsilon=1)
    model_SVR.fit(X_train, y_train)
    # y_pred = model_SVR.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(r2)
    models.append([model_KNN, model_DT, model_SVR])

    # 决策变量2      KNN，决策树，多层感知机
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y2, test_size=0.2, random_state=42)

    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
    # y_pred = model_KNN.predict(X_test)
    # r2 = r2_score(y_test, y_pred)
    # print(r2)

    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)
    # y_pred = model_DT.predict(X_test)
    # r2 = r2_score(y_test, y_pred)
    # print(r2)

    # 多层感知机
    # model_MLP = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
    # model_MLP.fit(X_train, y_train)
    # y_pred = model_MLP.predict(X_test)

    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(r2)
    models.append([model_KNN, model_DT, model_DT])

    # 决策变量3      线性回归，KNN，多层感知机
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y3, test_size=0.2, random_state=42)
    # 线性回归
    model_LR = LinearRegression()
    model_LR.fit(X_train, y_train)
    # y_pred = model_LR.predict(X_test)
    # r2 = r2_score(y_test, y_pred)
    # print(r2)
    # KNN
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
    # y_pred = model_KNN.predict(X_test)
    # r2 = r2_score(y_test, y_pred)
    # print(r2)

    # 多层感知机
    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)
    # y_pred = model_DT.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(r2)

    models.append([model_LR,model_KNN,model_DT])
    return models


def model_y(model_KNN,model_DT,model_RF,weight_y,X):
    # print(X)
    X = np.array(X).reshape(1,-1)
    result = weight_y[0] * model_KNN.predict(X) + weight_y[1] * model_DT.predict(X) + weight_y[2] * model_RF.predict(X)
    return result