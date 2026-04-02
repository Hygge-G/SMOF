import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

data = pd.read_csv('test_model_data随机性测试2.csv')
print(data)
x = range(1,len(data['平台利润'])+1)
y1 = data['企业利润']
y2 = data['平台利润']
y3 = data['消费者总效益']

# 创建图像，设置尺寸和分len()辨率（可选）
plt.figure(figsize=(8, 5), dpi=100)

# 绘制曲线（自定义颜色、线宽、标签）

plt.plot(x, y1,
         label='企业利润',
         color='orange',
         linestyle='-',  # 虚线样式
         linewidth=1)
plt.plot(x, y2,
         label='平台利润',
         color='red',
         linestyle='-',  # 虚线样式
         linewidth=1)
# plt.plot(x, y3,
#          label='消费者总效益',
#          color='blue',
#          linestyle='-',  # 虚线样式
#          linewidth=1)

# 添加标题和坐标轴标签
plt.title('检验', fontsize=14)
plt.xlabel('次数', fontsize=12)
plt.ylabel('值', fontsize=12)

# 添加图例并设置位置
plt.legend(loc='upper right', fontsize=10)

# 设置网格线
plt.grid(True,
         linestyle='-',
         alpha=0.4)

# 调整坐标轴范围
# plt.xlim(0, 2*np.pi)
# plt.ylim(-1.5, 1.5)

# 可选：设置坐标刻度


# 保存图像（需在plt.show()之前）
plt.savefig('平均过程.png', bbox_inches='tight')

# 显示图像
plt.show()

