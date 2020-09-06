"""
入口函数
"""
from model.lstm_choose_parameter import choose_parameter_run
from model.lstm_train import train_run
from model.lstm_predict import predict_run


def run():
    """

    :return:
    """
    file = './data/001632.csv'
    # 超参
    # choose_parameter_run(file=file)
    # train
    # train_run(file,
    #           unit=64,
    #           act='softsign',
    #           loss='mae',
    #           opt='RMSprop',
    #           l2_ratio=0)
    # predict
    predict_run(file=file)


if __name__ == '__main__':
    run()
