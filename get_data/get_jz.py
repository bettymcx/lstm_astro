"""
获取基金净值
https://finance.sina.com.cn/fund/
"""
import datetime
import numpy as np
import pandas as pd
import requests


def get_jz_data(code, start=None, end=None):
    """

    :param code:
    :param start:
    :param end:
    :return:
    """
    def get_single_page(code, start, end, page=1):
        """
        获取单页数据
        :param code:
        :param start:
        :param end:
        :param page:
        :return:
        """
        print('page: {}'.format(page))
        arr = []
        next_page = False

        url = 'https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/' \
              'CaihuiFundInfoService.getNav'

        parameter = {
            'symbol': code,
            'datefrom': start,
            'dateto': end,
            'page': page
        }

        header = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/84.0.4147.89 Safari/537.36'
        }

        req = requests.get(url=url, params=parameter, headers=header)
        if req.status_code == 200:
            json_data = req.json()

            rs_data = json_data.get('result').get('data') \
                if json_data.get('result') else None

            arr = rs_data.get('data') if rs_data else None
            total_size = rs_data.get('total_num') if rs_data else None
            if total_size:
                next_page = True \
                    if total_size and np.ceil(int(total_size) / 21) > page \
                    else False

        return arr, next_page

    start = start if start else \
        '{}-01-01'.format(int(datetime.datetime.now().strftime('%Y')) - 5)
    end = end if end else datetime.datetime.now().strftime('%Y-%m-%d')
    print('start: {}, end: {}'.format(start, end))

    arr = []
    page = 0
    next = True
    while next:
        page += 1
        mid_arr, next = get_single_page(code=code,
                                        start=start,
                                        end=end,
                                        page=page)
        if mid_arr:
            arr += mid_arr
        print('next: {}, page: {}'.format(next, page))

    if arr:
        df = pd.DataFrame(arr)
        if arr:
            df.columns = ['date', 'jz', 'ljjz']
            df['code'] = \
                pd.DataFrame([code for _ in range(len(df))], index=df.index)
            df['date'] = \
                df['date'].apply(lambda x:
                                 datetime.datetime.strptime(x[:10], '%Y-%m-%d'))

        df.to_csv('../data/{}.csv'.format(code))


if __name__ == '__main__':
    get_jz_data(code='001632')
