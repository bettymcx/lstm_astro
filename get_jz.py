import time
import pandas as pd
import requests


url = 'http://api.fund.eastmoney.com/f10/lsjz'
header = {
    # 'Referer': 'http://fundf10.eastmoney.com/jjjz_006328.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36'
}
parameter = {
    'callback': 'jQuery183023899070532104805_1599111573354',
    'fundCode': '006328',
    'pageIndex': 1,
    'pageSize': 50,
    'startDate': '2020-08-01',
    'endDate': '2020-09-03',
    '_': int(time.time()*1000)
}

url = 'http://api.fund.eastmoney.com/f10/lsjz?callback=jQuery183023899070532104805_1599111573354&fundCode=006328&pageIndex=1&pageSize=20&startDate=2018-08-01&endDate=2020-09-03&_={}'.format(int(time.time()*1000))

# req = requests.get(url=url, params=parameter, headers=header)
req = requests.get(url=url, headers=header)
print(req.status_code)
if req.status_code == 200:
    print(req.text)
