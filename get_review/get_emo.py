

# with open('mi_14默认排序_京东自营.csv', 'r', encoding='utf-8') as f:
#     lines = [line.strip() for line in f.readlines()]
# # print(lines)
# # 定位数据起始行（跳过文件头）
# start_index = 0
# while start_index < len(lines) and not lines[start_index].startswith("昵称"):
#     start_index += 1
# start_index += 1  # 跳过标题行

import pandas as pd
data = pd.read_csv('mi_15默认排序_京东自营.csv')
data = data.replace(r'\n',' ', regex=True)
print(data['输出'])
print(data['思考过程'])
print(data['评论数'])
# print(lines[1].split(','))
data.to_csv('mi_15默认排序_京东自营_清洗后.csv',encoding="utf_8_sig",index=False)
# 提取前10条评论（每6行一条完整记录）
# comments = []
# for idx in range(10):
#     line = lines[idx+1].split(',')
#     comment = {
#         "昵称": line[0],
#         "地区": line[1],
#         "产品": line[2],
#         "日期": line[3],
#         "评论内容": line[4],
#         "评分": line[5]
#     }
#     comments.append(comment)
# print('comment',comments)
# # 打印结果
# for i, cmt in enumerate(comments, 1):
#     print(f"【第{i}条评论】")
#     print(f"用户：{cmt['昵称']}（{cmt['地区']}）")
#     print(f"产品：{cmt['产品']}  时间：{cmt['日期']}")
#     print(f"评分：{cmt['评分']}星\n内容：{cmt['评论内容']}\n")

