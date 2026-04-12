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
chan_zhi = 'agency selling'
chan_zhuan = 'reselling'

review_1_1 = get_review_path('./data/data_review/mi_14.csv')
review_1_2 = get_review_path('./data/data_review/mi_15.csv')
review_2_1 = get_review_path('./data/data_review/mi_14.csv')
review_2_2 = get_review_path('./data/data_review/mi_15.csv')


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
            flt_phone,cur_phone = person.STEP1(platform,person)
            result_phone = None
            result_fit = 0
            if flt_phone:
                result_fit,result_phone = person.STEP2(flt_phone,cur_phone,platform,pis)

            if result_phone and random.random()>person.benefit:
                person.benefit = round(result_fit,2)
                person.buy(person,result_phone)

    f1 = A_prifit(platform)
    f2 = platform_prifit(platform)
    f3 = sum_benefit(customers)
    return f1,f2,f3


if __name__ == '__main__':
    f = open('sensitivity_result-sig_200.csv', 'w', newline='', encoding='utf-8-sig')
    fieldnames = [
    'Sensitive Variables', 'Scenario', 'Sensitive Variable Values',
    'Product Pricing', 'New Product Pricing', 'Product Wholesale Price', 'New Product Wholesale Price',
    'Product Retail Price', 'New Product Retail Price', 'Advertising Coefficient',
    'Corporate Profit', 'Platform Profit', 'Total Consumer Benefit', 'Time Consumption'
    ]
    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
    csv_writer.writeheader()
    mins = [1.0, 1.2, 0.9, 1.2, 0.9, 1.2, 0.0]
    maxs = [1.2, 1.5, 1.2, 1.5, 1.2, 1.5, 1.0]
    var_names = ['Product Pricing', 'New Product Pricing', 'Product Wholesale Price', 'New Product Wholesale Price',
    'Product Retail Price', 'New Product Retail Price', 'Advertising Coefficient']

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
                x[var_idx] = val   
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
                    'Sensitivity variables': var_names[var_idx],
                    'scenario': f'Q{scene_idx + 1}',
                    'sen_value': val,
                    'pa1': P_g[0],
                    'pa2': P_g[1],
                    'w1': Q_g[0],
                    'w2': Q_g[1],
                    'pr1': E_g[0],
                    'pr2': E_g[1],
                    'ad': ads,
                    'profitM': results[0],
                    'profitR': results[1],
                    'U': results[2],
                    'time_cost': use_time
                }
                csv_writer.writerow(row)

    f.close()

    # show
    for var_idx in range(7):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        titles = ['profitM', 'prifitR', 'U']

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
        fig.suptitle(f'sensitivity_analysis-{var_names[var_idx]}')
        save_path = f'image/sensitivity_result_{var_names[var_idx]}.png'
        fig.savefig(save_path, dpi=500)
        plt.tight_layout()
        plt.show()
