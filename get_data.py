import requests


def get_url(type, period, top_num=10):
    """

    :param type:
    :return:
    """
    url = 'https://fundapi.eastmoney.com/fundtradenew.aspx?ft={}&sc={}&st=desc&pi=1&pn={}&cp=&ct=&cd=&ms=&fr=&plevel=&fst=&ftype=&fr1=&fl=0&isab=1'.format(type, period, top_num)

    req = requests.get(url)
    if req.status_code == 200:
        return req.text


def parse_html(html):
    """

    :param html:
    :return:
    """
    html = html[html.index('[') + 1: html.index(']')]
    html = html.replace('"', '')
    arr = [data.split('|') for data in html.split(',')]
    return [data[0] for data in arr if len(data) >= 20]


def get_type_info(type, peroid_list, top_num=10):
    """

    :param type:
    :return:
    """
    code = set()
    for period in peroid_list:
        html = get_url(type=type, period=period, top_num=top_num)
        if html:
            codes = parse_html(html=html)
            code = code & set(codes) if code else set(codes)

    return code


def run():
    """

    :return:
    """
    type_list = ['gp', 'zs', 'qdii', 'hh']
    peroid_list = ['y', '3y', '6y', '1n']
    code = get_type_info(type='qdii', peroid_list=peroid_list, top_num=20)

    print(code)


if __name__ == '__main__':
    run()
