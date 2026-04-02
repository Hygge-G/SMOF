from function import *
import time
random.seed(1)
# 计时

# 决策变量
# 企业
P_ag = [1.2,1.3,1.5]         # 销售价格
P_bg = [1.2,1.3,1.5]
Q_ag = [1.1,1.2,1.3]         # 批发价格
Q_bg = [1.1,1.2,1.3]
ads = [0.2,0.3]
# 平台     六款产品的定价
E_ag = [1.2,1.3,1.5]
E_bg = [1.2,1.3,1.5]

# 初始参数
C_bg = [0.8,0.9,1.0]              # 产品成本
cusomer_num = 100000
# 厂商2、渠道2、产品2
# 定义产品多个层次的科技含量
tech_A_1 = [1.3037, 1.1576, 0.5519, 1.0406]# list(np.random.normal(1,0.2,4))        # 均值为1，方差为0.5的四个数
tech_A_2 = [0.7535, 1.1367, 0.6304, 1.1778]#list(np.random.normal(1,0.2,4))
tech_A_3 = [0.8535, 1.2367, 0.7304, 1.2778]#list(np.random.normal(1,0.2,4))
tech_B_1 = [1.1766, 0.6847, 0.7822, 1.3719]# list(np.random.normal(1,0.2,4))
tech_B_2 = [1.0639, 0.6114, 1.2182, 1.3073]# list(np.random.normal(1,0.2,4))
tech_B_3 = [1.1639, 0.7114, 1.3182, 1.6073]# list(np.random.normal(1,0.2,4))
k = 5                                      # 广告成本系数
per_day_inloss = 0.00064383   # 每天的手机损耗  0.00064383          51/12 = 4.25    1/4.25/365
pa1 = 1.1         # 企业A产品1的定价
pa2 = 1.2         # 企业A产品2的定价
pb1 = 1.1         # 企业B产品1的定价
pb2 = 1.2         # 企业B产品2的定价
paw1 = 0.8        # 企业A产品1的批发价
paw2 = 0.8        # 企业A产品2的批发价
pbw1 = 0.8        # 企业B产品1的批发价
pbw2 = 0.8        # 企业B产品2的批发价
recomA = 1.5      # 企业A对新品的广告投入系数
recomB = 1.5      # 企业B对新品的广告投入系数

# 初始参数



chan_zhi = '直销'
chan_zhuan = '转销'
# 企业定义
class company:
    def __init__(self,name,product1,product2,product3): # 两个产品
        self.name = name                  # 企业代号
        self.product1 = product1          # 企业中产品1
        self.product2 = product2          # 企业中产品2
        self.product3 = product3          # 企业中产品3

# 产品定义
class product:
    def __init__(self,firm,price,trade_price,cost_price,tech,G,recommend,review,chanl):
        self.firm = firm                  # 所属公司
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

    def STEP2(self,flt_phone,cur_phone):                # 广告推荐，群体效应对决策的影响，对于每一个产品而言
        cur_fit = []
        for i in range(len(cur_phone)):
            ifu_in = computer_inifu(cur_phone[i])    # 外部效应，销量得改一下
            ifu_out = computer_outifu(cur_phone[i],get_ano_phone(platform,cur_phone[i]))          # 内部效应，评论
            ifu_people = (1+ifu_in)*(1+ifu_out)
            flt_phone[i] *= ifu_people
            cur_fit.append(flt_phone[i])                # 更新后的产品效益

        result_fit = max(cur_fit)
        result = cur_phone[cur_fit.index(result_fit)]
        return result_fit,result            #返回最后购买的产品,和相对应的效益

    def buy(self,id,product):               # 购买行为，需要什么：渠道，产品
        customers[id].flag = 1              # 标记已经购买
        product.sell_num += 1               # 产品销量+1




path_mi = './data/data_mi1_emo.csv'
path_m2 = './data/data_mi_已获得最新时间评论_emo.csv'
review_A_1_1 = get_review_path(path_mi)
review_A_1_2 = get_review_path(path_mi)
review_A_1_3 = get_review_path(path_mi)
review_A_2_1 = get_review_path(path_mi)
review_A_2_2 = get_review_path(path_mi)
review_A_2_3 = get_review_path(path_mi)

review_B_1_1 = get_review_path(path_m2)
review_B_1_2 = get_review_path(path_m2)
review_B_1_3 = get_review_path(path_mi)
review_B_2_1 = get_review_path(path_m2)
review_B_2_2 = get_review_path(path_m2)
review_B_2_3 = get_review_path(path_mi)




F1 = []
F2 = []
for xx in range(3):       #循环10次了

    time_start_1 = time.time()

    A_chan1_1 = product('A', P_ag[1-1],trade_price=0,cost_price=C_bg[1-1],tech=tech_A_1,G=1,recommend=0       ,sell_num=0,review=review_A_1_1,chanl=chan_zhi)  # 公司，售价，批发价，技术含量，代数，广告推荐，销量、评论集   产品所在的厂商，渠道都是其属性
    A_chan1_2 = product('A', P_ag[2-1],trade_price=0,cost_price=C_bg[2-1],tech=tech_A_2,G=2,recommend=0       ,sell_num=0,review=review_A_1_2,chanl=chan_zhi)
    A_chan1_3 = product('A', P_ag[3-1],trade_price=0,cost_price=C_bg[3-1],tech=tech_A_3,G=3,recommend=ads[1-1],sell_num=0,review=review_A_1_3,chanl=chan_zhi)
    B_chan1_1 = product('B', P_bg[1-1],trade_price=0,cost_price=C_bg[1-1],tech=tech_B_1,G=1,recommend=0       ,sell_num=0,review=review_B_1_1,chanl=chan_zhi)
    B_chan1_2 = product('B', P_bg[2-1],trade_price=0,cost_price=C_bg[2-1],tech=tech_B_2,G=2,recommend=0       ,sell_num=0,review=review_B_1_2,chanl=chan_zhi)
    B_chan1_3 = product('B', P_bg[3-1],trade_price=0,cost_price=C_bg[3-1],tech=tech_B_3,G=3,recommend=ads[2-1],sell_num=0,review=review_B_1_3,chanl=chan_zhi)

    A_chan2_1 = product('A', E_ag[1-1],trade_price=Q_ag[1-1],cost_price=C_bg[1-1],tech=tech_A_1,G=1,recommend=0       ,sell_num=0,review=review_A_2_1,chanl=chan_zhuan)  # 公司，售价，批发价，技术含量，代数，广告推荐，销量、评论集   产品所在的厂商，渠道都是其属性
    A_chan2_2 = product('A', E_ag[2-1],trade_price=Q_ag[2-1],cost_price=C_bg[2-1],tech=tech_A_2,G=2,recommend=0       ,sell_num=0,review=review_A_2_2,chanl=chan_zhuan)
    A_chan2_3 = product('A', E_ag[3-1],trade_price=Q_ag[3-1],cost_price=C_bg[3-1],tech=tech_A_3,G=3,recommend=ads[1-1],sell_num=0,review=review_A_2_3,chanl=chan_zhuan)
    B_chan2_1 = product('B', E_bg[1-1],trade_price=Q_bg[1-1],cost_price=C_bg[1-1],tech=tech_B_1,G=1,recommend=0       ,sell_num=0,review=review_B_2_1,chanl=chan_zhuan)
    B_chan2_2 = product('B', E_bg[2-1],trade_price=Q_bg[2-1],cost_price=C_bg[2-1],tech=tech_B_2,G=2,recommend=0       ,sell_num=0,review=review_B_2_2,chanl=chan_zhuan)
    B_chan2_3 = product('B', E_bg[3-1],trade_price=Q_bg[3-1],cost_price=C_bg[3-1],tech=tech_B_3,G=3,recommend=ads[2-1],sell_num=0,review=review_B_2_3,chanl=chan_zhuan)
    # 创建产品集合
    channel_dai = [A_chan1_1, A_chan1_2,A_chan1_3, B_chan1_1, B_chan1_2,B_chan1_3]  # 换成列表
    channel_zhuan = [A_chan2_1, A_chan2_2, A_chan2_3, B_chan2_1, B_chan2_2,B_chan2_3]
    platform = [channel_dai, channel_zhuan]
    # 创建消费者群体
    customers = []
    for i in range(1,cusomer_num+1):
        id = i
        sen_price = round(np.random.normal(1, 0.3),2)
        sen_tech = list(np.random.dirichlet(np.ones(4)))           # 多个指标
        phone_loss = round(random.random(),2)
        temp = customer(id,sen_price,sen_tech,phone_loss)
        customers.append(temp)
    # 计算实验流程
    for day in range(365):
        for per in customers:
            per.benefit = max(per.benefit-per_day_inloss,0)                             # 用户当前产品效益每天都在下降
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
                result_fit,result_phone = person.STEP2(flt_phone,cur_phone)         # 输入是产品和效益，输出也是，只不过加上了群体效应和推荐影响

            if result_phone and random.random()>person.benefit:                                   # 临时放弃购买的概率
                person.benefit = round(result_fit,2)                             # 更新消费者效益
                person.buy(person.id,result_phone)                # 这个人在这一天购买了这个产品，效益为这个
                Print(person,result_phone,result_fit,day)                        # 打印结果


    '''
    目标函数：
    f1：企业A的利润
    f2：企业B的利润
    f3：平台的利润
    f4：消费者总效益
    '''
    f1 = A_prifit(platform)
    f2 = B_prifit(platform)
    f3 = platform_prifit(channel_zhuan)
    f4 = sum_benefit(customers)
    f5 = print_num(customers)
    F1.append(f1)
    F2.append(f2)
    print("企业A的销售量为：",f1)
    print("企业B的销售量为：",f2)
    print("平台总收益为：",f3)
    print("消费者总效益为：",round(f4,2))
    print("购买人数为{}，占总人数的{}%".format(f5,round(f5/len(customers)*100,2)))

    df = pd.DataFrame({'企业A的销售量': [f1],
                       '企业B的销售量': [f2],
                       '平台总收益': [f3],
                       '消费者总效益':[round(f4)],
                       '购买人数占比':[round(f5/len(customers)*100,2)]})
    df.to_csv('save_data2.csv', index=False)


    time_end_1 = time.time()
    print("运行时间："+str(round(time_end_1 - time_start_1,2))+"秒")
print('F1:',F1)
print('F2:',F2)