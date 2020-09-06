"""
数据获取&预处理
"""
import datetime
import pandas as pd


def get_data(file):
    """

    :param file:
    :return:
    """
    df = pd.read_csv(file)

    df = df[['date', 'ljjz']]
    df.columns = ['date', 'close']
    df = df.sort_values('date')

    df['pred'] = df['close'].shift(-1)
    df = df.iloc[: -1]

    df['date'] = \
        df['date'].apply(
            lambda x: datetime.datetime.strptime(x[: 10], '%Y-%m-%d'))
    df.set_index(keys='date', inplace=True)
    return df


def preprocess(train_data, valid_data, test_data):
    """
    预处理：reshape为目标shape
    :param train_data:
    :param valid_data:
    :param test_data:
    :return:
    """
    train_x = train_data[:, : -1]
    train_y = train_data[:, -1:]
    valid_x = valid_data[:, : -1]
    valid_y = valid_data[:, -1:]
    test_x = test_data[:, : -1]
    test_y = test_data[:, -1:]

    # 每日
    step = 1
    train_x = train_x.reshape((train_x.shape[0], step, train_x.shape[1]))
    train_y = train_y.reshape((train_y.shape[0], step, train_y.shape[1]))
    valid_x = valid_x.reshape((valid_x.shape[0], step, valid_x.shape[1]))
    valid_y = valid_y.reshape((valid_y.shape[0], step, valid_y.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], step, test_x.shape[1]))
    test_y = test_y.reshape((test_y.shape[0], step, test_y.shape[1]))

    return train_x, train_y, valid_x, valid_y, test_x, test_y
