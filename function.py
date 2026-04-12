'''
Define various required functions
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


class company:
    def __init__(self,name,product1,product2,product3): 
        self.name = name                 
        self.product1 = product1          
        self.product2 = product2        
        self.product3 = product3        


class product:
    def __init__(self,price,trade_price,cost_price,tech,G,recommend,review,chanl):
        self.price = price                
        self.trade_price = trade_price   
        self.cost_price = cost_price      
        self.tech = tech                  
        self.G = G                    
        self.recommend = recommend     
        self.sell_num = 0             
        self.review = review          
        self.chanl = chanl      

class comment:                     
    def __int__(self,id,time,emo):
        self.id = id           
        self.time = time       
        self.emo = emo           


class channel:
    def __init__(self,productA1,productA2,productA3,productB1,productB2,productB3):     
        self.productA1 = productA1              
        self.productA2 = productA2             
        self.productA3 = productA3            
        self.productB1 = productB1               
        self.productB2 = productB2              
        self.productB3 = productB3                 



class customer:
    def __init__(self,id,sen_price,sen_tech,benefit):
        self.flag = 0           
        self.id = id                
        self.sen_price = sen_price    
        self.sen_tech = sen_tech     
        self.benefit = benefit       

    def STEP1(self,platform,person):    
        cur_phone = []
        flt_phone = []
        for chan in platform:              
            for shouji in chan:           
                key = computer_fit(shouji,person.sen_price,person.sen_tech,person.benefit)   
                if key:
                    flt_phone.append(key)
                    cur_phone.append(shouji)   
        return flt_phone,cur_phone

    def STEP2(self,flt_phone,cur_phone,platform,pis):      
        cur_fit = []
        for i in range(len(cur_phone)):
            ifu_in = computer_inifu(cur_phone[i],pis) 

            ifu_people = (1+ifu_in-0.5)
            flt_phone[i] *= ifu_people * (1+cur_phone[i].recommend)    
            cur_fit.append(flt_phone[i])         
        result_fit = max(cur_fit)
        result = cur_phone[cur_fit.index(result_fit)]
        return result_fit,result         

    def buy(self,person,product):          
        person.flag = 1        
        product.sell_num += 1       


def computer_inifu(phone,pis):       
    comment = phone.review
    emo = random.choices(comment,list(pis),k=1)
    if isinstance(emo[0],str):
        return float(emo[0].strip())
    else:
        return emo[0]
def computer_outifu(phone1,phone2):
    n1 = phone1.sell_num
    n2 = phone2.sell_num
    return (n1)/(n1+n2+0.0000001)  

def get_ano_phone(platform,cur_phone):
    for chan in range(len(platform)):
        for phone in range(len(platform[chan])):
            if platform[chan][phone] == cur_phone:
                return platform[not chan][phone]
    print("找不到对应渠道的产品，产品有改动")
    return None

def computer_fit(shouji,sen_price,sen_tech,benefit):    
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
    temp = data['output']
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


def Print_cumsomers(customers):
    for per in customers:
        print(No.{0} user".format(per.id))
        print('技术敏感度：',per.sen_tech)
        print('价格敏感度：', per.sen_price)
        print('手机损耗度：', per.phone_loss)


def A_prifit(platform):      
    sumA = 0

    for mi in platform[0]:
        sumA += mi.sell_num * (mi.price - mi.cost_price)     
        sumA -= mi.sell_num * mi.price * 0.036     
    for mi in platform[1]:
        sumA += mi.sell_num * (mi.trade_price - mi.cost_price)    
    sumA -= ((mi.recommend)*(mi.recommend)*1000)/2  
    return round(sumA,2)


def platform_prifit(platform):  
    sumC = 0
    for per_product in platform[0]:
        sumC += per_product.sell_num * per_product.price * 0.036
    for per_product in platform[1]:
        sumC += per_product.sell_num * (per_product.price - per_product.trade_price)
    return round(sumC,2)

def sum_benefit(customers):       
    sum_fit = 0
    for person in customers:
        if person.flag==1:
            sum_fit += person.benefit
    return round(sum_fit,2)
def print_num(customers):     
    x = 0
    for i in customers:
        x += i.flag
    return x


def get_model(normalized_X,y1,y2,y3):
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
    models.append([model_KNN, model_DT, model_DT])
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y3, test_size=0.2, random_state=42)

    model_LR = LinearRegression()
    model_LR.fit(X_train, y_train)
    model_KNN = KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
    model_DT = DecisionTreeRegressor(random_state=42)
    model_DT.fit(X_train, y_train)
    models.append([model_LR,model_KNN,model_DT])
    return models


def model_y(model_KNN,model_DT,model_RF,weight_y,X):
    X = np.array(X).reshape(1,-1)
    result = weight_y[0] * model_KNN.predict(X) + weight_y[1] * model_DT.predict(X) + weight_y[2] * model_RF.predict(X)
    return result