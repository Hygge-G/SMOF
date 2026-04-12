# load
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

dp = ChromiumPage()

dp.listen.start('https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0')
dp.get('https://item.jd.com/10089389490404.html')
dp.scroll.to_bottom()
# 点击current
dp.ele('css:.current').click()
flag = 0

time.sleep(2)

for i in range(1,101):
    print('正在爬取第{}页'.format(i))
    resp = dp.listen.wait()
    json_data = resp.response.body
    dp.ele('css:.current').click()
    dp.scroll.to_bottom()

    comments = json_data['comments']
    print(comments)
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
    if i!=100:
        dp.ele('css:ui-pager-next').click()
        time.sleep(random.randint(1, 5))  # 随机暂停几秒

