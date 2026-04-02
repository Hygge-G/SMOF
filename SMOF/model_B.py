import random
import pandas as pd
from function import *
import time
import csv
random.seed(1)

# 固定参数
cusomer_num = 100000
# 厂商2、渠道2、产品2
# 定义产品多个层次的科技含量
tech_A_1 = [1.02263626, 0.94142316, 1.00204377, 0.8959569 ]#[1.3037, 1.1576, 0.5519, 1.0406]# list(np.random.normal(1,0.2,4))        # 均值为1，方差为0.5的四个数
tech_A_2 = [1.10177075, 1.00801234, 1.10905282, 1.10346105]#[0.7535, 1.1367, 0.6304, 1.1778]#list(np.random.normal(1,0.2,4))
                                 # 广告成本系数
per_day_inloss = 0.00064383   # 每天的手机损耗  0.00064383          51/12 = 4.25    1/4.25/365

chan_zhi = '直销'
chan_zhuan = '转销'
# 评论内容
# 渠道 代数
review_1_1 = get_review_path('./data/data_review/mi_14默认排序_官方直营.csv')
review_1_2 = get_review_path('./data/data_review/mi_15默认排序_官方直营.csv')

review_2_1 = get_review_path('./data/data_review/mi_14默认排序_京东自营.csv')
review_2_2 = get_review_path('./data/data_review/mi_15默认排序_京东自营.csv')




def forge_platform_customer(P_g,E_g,Q_g,C_g,ads):
    # 创建产品对象
    # 渠道1
    chan1_1 = product(P_g[1-1],trade_price=0,cost_price=C_g[1-1],tech=tech_A_1,G=1,recommend=0       ,review=review_1_1,chanl=chan_zhi)  # 公司，售价，批发价，技术含量，代数，广告推荐，销量、评论集   产品所在的厂商，渠道都是其属性
    chan1_2 = product(P_g[2-1],trade_price=0,cost_price=C_g[2-1],tech=tech_A_2,G=2,recommend=ads,review=review_1_2,chanl=chan_zhi)

    # 渠道2
    chan2_1 = product(E_g[1-1],trade_price=Q_g[1-1],cost_price=C_g[1-1],tech=tech_A_1,G=1,recommend=0       ,review=review_2_1,chanl=chan_zhuan)  # 公司，售价，批发价，技术含量，代数，广告推荐，销量、评论集   产品所在的厂商，渠道都是其属性
    chan2_2 = product(E_g[2-1],trade_price=Q_g[2-1],cost_price=C_g[2-1],tech=tech_A_2,G=2,recommend=ads,review=review_2_2,chanl=chan_zhuan)


    # 创建产品集合
    channel_dai = [chan1_1, chan1_2]
    channel_zhuan = [chan2_1, chan2_2]
    # 创建平台
    platform = [channel_dai, channel_zhuan]
    # 创建消费者群体
    customers = []
    for i in range(1,cusomer_num+1):
        id = i
        sen_price = round(np.random.normal(0.8, 0.45),2)
        sen_tech = list(np.random.dirichlet(np.ones(4)) * np.random.normal(1,0.15))           # 多个指标
        phone_loss = round(random.random(),2)
        temp = customer(id,sen_price,sen_tech,phone_loss)
        customers.append(temp)

    return platform,customers

def model(platform,customers,pis):
    # 计算实验流程
    for day in range(365):
        for per in customers:
            per.benefit -= per_day_inloss
            per.benefit = max(per.benefit,0)                             # 用户当前产品效益每天都在下降
        customer_day = random.sample(customers, len(customers)//30)      # 每次选择3%的用户
        for person in customer_day:                                 # 在曝光中选择用户
            if person.flag == 1:                                    # 已购买用户不再重复购买
                # print("用户{0:<5}已购买，不再重新购买".format(person.id))      #格式控制
                continue
            # 进行第一步决策   返回效益集合与产品集合
            flt_phone,cur_phone = person.STEP1(platform,person)       #输入消费者和产品集　返回的应该是符合条件的产品与其对应的效益
            result_phone = None
            result_fit = 0
            if flt_phone:
                result_fit,result_phone = person.STEP2(flt_phone,cur_phone,platform,pis)         # 输入是产品和效益，输出也是，只不过加上了群体效应和推荐影响

            if result_phone and random.random()>person.benefit:                                   # 临时放弃购买的概率
                person.benefit = round(result_fit,2)                             # 更新消费者效益
                person.buy(person,result_phone)                # 这个人在这一天购买了这个产品，效益为这个
                # Print(person,result_phone,result_fit,day)                        # 打印结果

    '''
    目标函数：
    f1：企业利润
    f3：平台利润
    f4：消费者总效益
    '''
    f1 = A_prifit(platform)
    f2 = platform_prifit(platform)  # channel_zhuan[1]
    f3 = sum_benefit(customers)


    return f1,f2,f3

if __name__ == '__main__':
    f = open('model_data_纠正消费者效益-去除随机种子-test-graph.csv', mode='w', encoding='utf-8-sig', newline='')
    csv_writer = csv.DictWriter(f, fieldnames=[
        '产品定价',
        '新品定价',
        '产品批发价',
        '新品批发价',
        '产品销售价',
        '新品销售价',
        '广告系数',
        '企业利润',
        '平台利润',
        '消费者总效益',
        '耗时'
    ])
    csv_writer.writeheader()
    # 决策变量
    # 企业
    P_g = [1.2, 1.3]  # 销售价格
    P_g = [1.2, 1.3]
    Q_g = [1.1, 1.2]  # 批发价格
    Q_g = [1.1, 1.2]

    # 平台     六款产品的定价

    E_g = [1.2, 1.3]
    # 初始参数
    #C_g = [1]  # 产品成本

    data = []
    for xx in range(20):
        random.seed(1)
        print('第'+ str(xx+1) +'组数据生成：')
        # 企业定价
        P_g[0] = random.uniform(1.0, 1.2)
        P_g[1] = random.uniform(1.2, 1.5)
        # 企业给平台的批发价，不能高于企业定价
        Q_g[0] = random.uniform(0.9,P_g[0]-random.uniform(0,0.1))
        Q_g[1] = random.uniform(1.0,P_g[1]-random.uniform(0,0.1))
        # 平台定价，应当高于批发价
        E_g[0] = random.uniform(Q_g[0]+random.uniform(0,0.1),1.25+random.uniform(0,0.1))
        E_g[1] = random.uniform(Q_g[1]+random.uniform(0,0.1),1.55+random.uniform(0,0.1))
        # 成本价格
        C_g = [0.9,1.0]
        # 广告投入
        ads = random.uniform(0,1)      # 企业A的广告投入

        time_start = time.time()
        platform,customers = forge_platform_customer(P_g,E_g,Q_g,C_g,ads)
        result = model(platform,customers,get_pi(200))
        time_end = time.time()
        use_time = str(round(time_end - time_start,2))+"秒"
        results = list(result)
        # 提取具体数据
        dit = {
            '产品定价':P_g[0],
            '新品定价':P_g[1],
            '产品批发价':Q_g[0],
            '新品批发价':Q_g[1],
            '产品销售价':E_g[0],
            '新品销售价':E_g[1],
            '广告系数':ads,
            '企业利润':results[0],
            '平台利润':results[1],
            '消费者总效益':results[2],
            '耗时':use_time
        }
        csv_writer.writerow(dit)
        for chan in platform:
            for mi in chan:
                print(mi.sell_num)
