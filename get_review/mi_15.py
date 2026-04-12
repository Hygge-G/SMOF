# 导入
from DrissionPage import ChromiumPage
import csv
import time
import random
f = open('data.csv',mode='w',encoding='utf-8-sig',newline='')

csv_writer = csv.DictWriter(f,fieldnames=[
    '昵称',
    '地区',
    '产品',
    '日期',
    '评论',
    '评分',
    '点赞量',
])
csv_writer.writeheader()
# 创建对象
dp = ChromiumPage()
dp.listen.start('https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc')  # &client=pc&clientVersion=1.0.0           https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments
dp.get('https://item.jd.com/100149708958.html')
dp.scroll.to_bottom()
# 点击current

dp.ele('css:.current').click()  # 点击商品的评论
flag = 1              # 排序方式，0是默认排序， 1是按照时间排序

dp.ele('css:.J-current-sortType').click()
temp = dp.eles('@class:J-sortType-item')  # 查找所有 class 含有 ele_class 的元素
temp[flag].click()  # 拿到自己需要的那个，即按照时间排序

time.sleep(2)  # 查看一下
# 监听
resp = dp.listen.wait()
json_data = resp.response.body

for i in range(1,101):
    print('正在爬取第{}页'.format(i))
    resp = dp.listen.wait()
    json_data = resp.response.body
    dp.scroll.to_bottom()


    comments = json_data['comments']
    for index in comments:

        dit = {
            '昵称':index['nickname'],
            '地区':index['location'],
            '产品':index['productColor'],
            '日期':index['creationTime'],
            '评论':index['content'].replace("\n", ""),
            '评分':index['score'],
            '点赞量':index['usefulVoteCount']
        }
        csv_writer.writerow(dit)
        print(dit)
    if i!=100:
        dp.ele('css:ui-pager-next').click()
        time.sleep(random.randint(1, 5))  #


