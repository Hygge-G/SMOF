import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


import matplotlib.pyplot as plt

from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 10.5,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)



# 读取 CSV 文件
file_path = r'E:\User\手机定价\sensitivity_result-sig_200.csv'
data = pd.read_csv(file_path)

# 数据整理：按“敏感性变量”和“场景”分组
grouped_data = data.groupby(['敏感性变量', '场景'])

# 获取所有的敏感性变量名称
var_names = data['敏感性变量'].unique()
from matplotlib import font_manager

# font = font_manager.FontProperties(family='Times New Roman', size=10.5)

var = ['p$_{A1}$','p$_{A2}$','w$_1$','w$_2$','P$_{R1}$','P$_{R2}$','d']
# 绘图
for var_idx in range(len(var_names)):
    # 创建图形和子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    titles = ['${Profit}_M$', '${Profit}_P$', '${Utility}$']

    # 获取当前敏感性变量的数据
    for scene_idx in range(3):
        scene_label = f'Q{scene_idx + 1}'

        # 获取当前场景的数据
        scene_data = grouped_data.get_group((var_names[var_idx], scene_label))
        # 绘制各个场景的折线图
        axes[0].plot(scene_data['敏感性变量取值'], scene_data['企业利润'], label=scene_label)
        axes[0].legend(loc='upper left')
        axes[1].plot(scene_data['敏感性变量取值'], scene_data['平台利润'], label=scene_label)
        axes[1].legend(loc='upper left')
        axes[2].plot(scene_data['敏感性变量取值'], scene_data['消费者总效益'], label=scene_label)
        axes[2].legend(loc='upper left')

    # 设置各个子图的标题和标签
    for j in range(3):
        axes[j].set_xlabel(var[var_idx])
        axes[j].set_ylabel(titles[j])
        axes[j].legend()
        axes[j].grid(False)

    # 设置整体标题
    # fig.suptitle(f'sensitivity analysis - {var[var_idx]}',fontproperties=font)

    # 保存图像
    save_path = f'data_img/sensitivity_result_{var_names[var_idx]}.png'  # 保存路径

    fig.savefig(save_path, dpi=500,bbox_inches='tight')  # 设置高分辨率

    # 自动调整布局并显示
    plt.tight_layout()
    plt.show()

