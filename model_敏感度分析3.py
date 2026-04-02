from function import *
import time
import csv
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

cusomer_num = 100000

tech_A_1 = [1.02263626, 0.94142316, 1.00204377, 0.8959569 ]
tech_A_2 = [1.10177075, 1.00801234, 1.10905282, 1.10346105]

per_day_inloss = 0.00064383

chan_zhi = '直销'
chan_zhuan = '转销'

review_1_1 = get_review_path('./data/data_review/mi_14默认排序_官方直营.csv')
review_1_2 = get_review_path('./data/data_review/mi_15默认排序_官方直营.csv')

review_2_1 = get_review_path('./data/data_review/mi_14默认排序_京东自营.csv')
review_2_2 = get_review_path('./data/data_review/mi_15默认排序_京东自营.csv')




def forge_platform_customer(P_g,E_g,Q_g,C_g,ads):

    chan1_1 = product(P_g[1-1],trade_price=0,cost_price=C_g[1-1],tech=tech_A_1,G=1,recommend=0       ,review=review_1_1,chanl=chan_zhi)
    chan1_2 = product(P_g[2-1],trade_price=0,cost_price=C_g[2-1],tech=tech_A_2,G=2,recommend=ads,review=review_1_2,chanl=chan_zhi)

    chan2_1 = product(E_g[1-1],trade_price=Q_g[1-1],cost_price=C_g[1-1],tech=tech_A_1,G=1,recommend=0       ,review=review_2_1,chanl=chan_zhuan)
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
            per.benefit = max(per.benefit,0)
        customer_day = random.sample(customers, len(customers)//30)
        for person in customer_day:
            if person.flag == 1:
                continue
            flt_phone,cur_phone = person.STEP1(platform,person)
            result_phone = None
            result_fit = 0
            if flt_phone:
                result_fit,result_phone = person.STEP2(flt_phone,cur_phone,platform,pis)

            if result_phone and random.random()>person.benefit:
                person.benefit = round(result_fit,2)
                person.buy(person,result_phone)
    '''
    目标函数：
    f1：企业利润
    f3：平台利润
    f4：消费者总效益
    '''
    f1 = A_prifit(platform)
    f2 = platform_prifit(platform)
    f3 = sum_benefit(customers)
    return f1,f2,f3







if __name__ == '__main__':

    f = open('sensitivity_result-sig_200.csv', 'w', newline='', encoding='utf-8-sig')
    fieldnames = [
        '敏感性变量', '场景', '敏感性变量取值',
        '产品定价', '新品定价', '产品批发价', '新品批发价',
        '产品销售价', '新品销售价', '广告系数',
        '企业利润', '平台利润', '消费者总效益', '耗时'
    ]
    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
    csv_writer.writeheader()

    mins = [1.0, 1.2, 0.9, 1.2, 0.9, 1.2, 0.0]
    maxs = [1.2, 1.5, 1.2, 1.5, 1.2, 1.5, 1.0]

    var_names = ['产品定价', '新品定价', '产品批发价', '新品批发价',
                 '产品销售价', '新品销售价', '广告系数']

    #  Q1/Q2/Q3
    Q1, Q2, Q3 = [], [], []
    for i in range(7):
        span = maxs[i] - mins[i]
        Q1.append(mins[i] + 0.25 * span)
        Q2.append(mins[i] + 0.50 * span)
        Q3.append(mins[i] + 0.75 * span)

    Q_all = [Q1, Q2, Q3]

    P_g = [0.0, 0.0]
    Q_g = [0.0, 0.0]
    E_g = [0.0, 0.0]
    C_g = [0.9, 1.0]

    # 每个变量扫描多少个点
    N_STEPS = 200

    sens_data = {}
    for var_idx in range(7):
        for scene_idx in range(3):
            sens_data[(var_idx, scene_idx)] = {
                'x': [],
                'f1': [],
                'f2': [],
                'f3': []
            }


    for var_idx in range(7):
        for scene_idx in range(3):
            print('var_idx,scene_idx',var_idx,scene_idx)
            base_vals = [Q_all[scene_idx][i] for i in range(7)]
            vmin = mins[var_idx]
            vmax = maxs[var_idx]
            for step in range(N_STEPS):
                val = vmin + (vmax - vmin) * step / (N_STEPS - 1)

                x = base_vals[:]
                x[var_idx] = val      #
                P_g[0] = x[0]
                P_g[1] = x[1]
                Q_g[0] = x[2]
                Q_g[1] = x[3]
                E_g[0] = x[4]
                E_g[1] = x[5]
                ads = x[6]
                random.seed(1)

                time_start = time.time()
                platform, customers = forge_platform_customer(P_g, E_g, Q_g, C_g, ads)
                result = model(platform, customers, get_pi(200))
                time_end = time.time()

                results = list(result)
                use_time = round(time_end - time_start, 4)

                d = sens_data[(var_idx, scene_idx)]
                d['x'].append(val)
                d['f1'].append(results[0])
                d['f2'].append(results[1])
                d['f3'].append(results[2])

                row = {
                    '敏感性变量': var_names[var_idx],
                    '场景': f'Q{scene_idx + 1}',
                    '敏感性变量取值': val,
                    '产品定价': P_g[0],
                    '新品定价': P_g[1],
                    '产品批发价': Q_g[0],
                    '新品批发价': Q_g[1],
                    '产品销售价': E_g[0],
                    '新品销售价': E_g[1],
                    '广告系数': ads,
                    '企业利润': results[0],
                    '平台利润': results[1],
                    '消费者总效益': results[2],
                    '耗时': use_time
                }
                csv_writer.writerow(row)

    f.close()

    # 可视化

    for var_idx in range(7):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        titles = ['企业利润', '平台利润', '消费者总效益']

        for scene_idx in range(3):
            label = f'Q{scene_idx + 1}'
            d = sens_data[(var_idx, scene_idx)]
            axes[0].plot(d['x'], d['f1'], label=label)
            axes[1].plot(d['x'], d['f2'], label=label)
            axes[2].plot(d['x'], d['f3'], label=label)
        for j in range(3):
            axes[j].set_xlabel(var_names[var_idx])
            axes[j].set_ylabel(titles[j])
            axes[j].legend()
            axes[j].grid(True)
        fig.suptitle(f'敏感性分析 - {var_names[var_idx]}')
        save_path = f'image/sensitivity_result_{var_names[var_idx]}.png'
        fig.savefig(save_path, dpi=500)
        plt.tight_layout()
        plt.show()
