import random
import pandas as pd
from function import *
import time
import csv
random.seed(1)

# Fixed parameters
cusomer_num = 100000
tech_A_1 = [1.02263626, 0.94142316, 1.00204377, 0.8959569 ]  # list(np.random.normal(1,0.2,4))     
tech_A_2 = [1.10177075, 1.00801234, 1.10905282, 1.10346105]  # list(np.random.normal(1,0.2,4))
per_day_inloss = 0.00064383   #          51/12 = 4.25    1/4.25/365

chan_zhi = 'agency selling'
chan_zhuan = 'reselling'
review_1_1 = get_review_path('./data/data_review/.csv')
review_1_2 = get_review_path('./data/data_review/.csv')

review_2_1 = get_review_path('./data/data_review/.csv')
review_2_2 = get_review_path('./data/data_review/.csv')



def forge_platform_customer(P_g,E_g,Q_g,C_g,ads):

    chan1_1 = product(P_g[1-1],trade_price=0,cost_price=C_g[1-1],tech=tech_A_1,G=1,recommend=0       ,review=review_1_1,chanl=chan_zhi)  
    chan1_2 = product(P_g[2-1],trade_price=0,cost_price=C_g[2-1],tech=tech_A_2,G=2,recommend=ads,review=review_1_2,chanl=chan_zhi)

    chan2_1 = product(E_g[1-1],trade_price=Q_g[1-1],cost_price=C_g[1-1],tech=tech_A_1,G=1,recommend=0       ,review=review_2_1,chanl=chan_zhuan) 
    chan2_2 = product(E_g[2-1],trade_price=Q_g[2-1],cost_price=C_g[2-1],tech=tech_A_2,G=2,recommend=ads,review=review_2_2,chanl=chan_zhuan)



    channel_dai = [chan1_1, chan1_2]
    channel_zhuan = [chan2_1, chan2_2]
    platform = [channel_dai, channel_zhuan]
    customers = []
    for i in range(1,cusomer_num+1):
        id = i
        sen_price = round(np.random.normal(0.8, 0.45),2)
        sen_tech = list(np.random.dirichlet(np.ones(4)) * np.random.normal(1,0.15))    
        phone_loss = round(random.random(),2)
        temp = customer(id,sen_price,sen_tech,phone_loss)
        customers.append(temp)
    return platform,customers

def model(platform,customers,pis):
    for day in range(365):
        for per in customers:
            per.benefit -= per_day_inloss
            per.benefit = max(per.benefit,0)      
        customer_day = random.sample(customers, len(customers)//30)   
        for person in customer_day:          
            if person.flag == 1:          
                continue
            result_phone = None
            result_fit = 0
            if flt_phone:
                result_fit,result_phone = person.STEP2(flt_phone,cur_phone,platform,pis)         

            if result_phone and random.random()>person.benefit:                              
                person.benefit = round(result_fit,2)               
                person.buy(person,result_phone)               


    '''
    obj：
    f1：profitM
    f3：profitR
    f4：M
    '''
    f1 = A_prifit(platform)
    f2 = platform_prifit(platform)
    f3 = sum_benefit(customers)


    return f1,f2,f3

if __name__ == '__main__':
    f = open('model_data.csv', mode='w', encoding='utf-8-sig', newline='')
    csv_writer = csv.DictWriter(f, fieldnames=[
        'pa1',
        'pa2',
        'w1',
        'w2',
        'pr1',
        'pr2',
        'd',
        'profitm',
        'profitr',
        'U',
        'time_cost'
    ])
    csv_writer.writeheader()
    P_g = [1.2, 1.3]  
    P_g = [1.2, 1.3]
    Q_g = [1.1, 1.2]  
    Q_g = [1.1, 1.2]
    E_g = [1.2, 1.3]

    data = []
    for xx in range(20):
        random.seed(1)
        print('No'+ str(xx+1) +'data：')
        P_g[0] = random.uniform(1.0, 1.2)
        P_g[1] = random.uniform(1.2, 1.5)
        Q_g[0] = random.uniform(0.9,P_g[0]-random.uniform(0,0.1))
        Q_g[1] = random.uniform(1.0,P_g[1]-random.uniform(0,0.1))
        E_g[0] = random.uniform(Q_g[0]+random.uniform(0,0.1),1.25+random.uniform(0,0.1))
        E_g[1] = random.uniform(Q_g[1]+random.uniform(0,0.1),1.55+random.uniform(0,0.1))
        C_g = [0.9,1.0]
        ads = random.uniform(0,1)   
        time_start = time.time()
        platform,customers = forge_platform_customer(P_g,E_g,Q_g,C_g,ads)
        result = model(platform,customers,get_pi(200))
        time_end = time.time()
        use_time = str(round(time_end - time_start,2))+"s"
        results = list(result)
   
        dit = {
            'pa1':P_g[0],
            'pa2':P_g[1],
            'w1':Q_g[0],
            'w2':Q_g[1],
            'pr1':E_g[0],
            'pr2':E_g[1],
            'd':ads,
            'profitm':results[0],
            'profitr':results[1],
            'U':results[2],
            'time_cost':use_time
        }
        csv_writer.writerow(dit)
        for chan in platform:
            for mi in chan:
                print(mi.sell_num)
