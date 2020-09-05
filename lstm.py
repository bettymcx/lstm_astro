"""
LSTM模型预测股票未来一个月的趋势
"""
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import L2
from keras.metrics import (RootMeanSquaredError,
                           MeanAbsoluteError,
                           MeanAbsolutePercentageError)
from pylab import mpl

# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False


def get_data(file):
    """

    :param file:
    :param valid_size:
    :param test_size:
    :return:
    """
    df = pd.read_csv(file)

    df = df[['date', 'ljjz']]
    df.columns = ['date', 'close']
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


def plot_info(train, test, title, ylabel):
    """
    训练 & 测试
    :param train:
    :param test:
    :param title:
    :param ylabel:
    :return:
    """
    plt.clf()

    plt.plot(train)
    plt.plot(test)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(['训练', '测试'], loc='upper left')

    plt.show()


def build_fit_model(train_x, train_y, valid_x, valid_y, test_x, test_y,
                    unit, act, loss, opt, l2_ratio,
                    epochs, batch_size,
                    verbose=1, workers=4, use_multiprocessing=True, plot=False):
    """

    :param train_x:
    :param train_y:
    :param valid_x:
    :param valid_y:
    :param test_x:
    :param test_y:
    :param unit: LSTM输出维度/神经网络数量
    :param act: 激活函数
    :param opt: 优化函数
    :param l2_ratio: L2正则化值
    :param loss: 损失函数
    :param epochs: 迭代次数
    :param batch_size:
    :param verbose: 训练中是否输出：1：输出；0：不输出
    :param workers: CPU核心数
    :param use_multiprocessing: 是否并发
    :param plot: 是否画出
    :return:
    """
    model = Sequential()
    model.add(LSTM(units=unit,
                   input_shape=(train_x.shape[1], train_x.shape[2]),
                   activation=act,
                   kernel_regularizer=L2(l2_ratio)))
    model.add(Dense(units=1))
    model.compile(loss=loss, optimizer=opt)

    # fit
    history = model.fit(train_x, train_y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(valid_x, valid_y),
                        verbose=verbose,
                        shuffle=False,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing)

    loss_metrics = model.evaluate(test_x, test_y,
                                  verbose=0,
                                  batch_size=batch_size)

    if plot:
        # loss
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plot_info(train=train_loss, test=val_loss,
                  title='LSTM 模型损失值',
                  ylabel='Loss', pic_path=None)

        # acc
        if 'acc' in history.history.keys():
            train_acc = history.history['acc']
            test_val_acc = history.history['val_acc']
            plot_info(train=train_acc, test=test_val_acc,
                      title='LSTM 模型准确率',
                      ylabel='Accuracy', pic_path=None)
    return model, loss_metrics


def grid_search(train_x, train_y, valid_x, valid_y, test_x, test_y,
                epochs, batch_size, units_arr, activate_arr, opt_arr, loss,
                reg_arr, workers):
    """
    对参数进行暴力搜素
    :param train_x:
    :param train_y:
    :param valid_x:
    :param valid_y:
    :param test_x:
    :param test_y:
    :param epochs:
    :param batch_size:
    :param units_arr:
    :param activate_arr:
    :param opt_arr:
    :param loss:
    :param reg_arr:
    :param workers:
    :return:
    """
    loss_value = 100000
    target_unit = None
    target_act = None
    target_opt = None
    target_reg = None

    for unit in units_arr:
        for opt in opt_arr:
            for act in activate_arr:
                for reg in reg_arr:
                    print('-' * 10)
                    model, loss_metrics = \
                        build_fit_model(train_x, train_y,
                                        valid_x, valid_y,
                                        test_x, test_y,
                                        unit=unit,
                                        act=act,
                                        loss=loss,
                                        opt=opt,
                                        epochs=epochs,
                                        l2_ratio=reg,
                                        batch_size=batch_size,
                                        verbose=0,
                                        workers=workers)
                    if loss_metrics < loss_value:
                        loss_value = loss_metrics
                        target_unit = unit
                        target_act = act
                        target_opt = opt
                        target_reg = reg

    info = 'unit: {}, act: {}, opt: {}, loss: {}, target_reg: {}, ' \
           'loss_meterics； {}'.format(target_unit, target_act, target_opt,
                                      loss, target_reg, loss_value)
    print(info)


def evalute(valid_y_true, pred_valid_y, test_y_true, pred_test_y):
    """

    :param valid_y_true:
    :param pred_valid_y:
    :param test_y_true:
    :param pred_test_y:
    :return:
    """
    # 评估
    # mae
    mae = MeanAbsoluteError()
    mae.update_state(y_true=valid_y_true, y_pred=pred_valid_y)
    valid_mae = mae.result()

    mae.reset_states()
    mae.update_state(y_true=test_y_true, y_pred=pred_test_y)
    test_mae = mae.result()

    # mape
    mape = MeanAbsolutePercentageError()
    mape.update_state(y_true=valid_y_true, y_pred=pred_valid_y)
    valid_mape = mape.result()

    mape.reset_states()
    mape.update_state(y_true=test_y_true, y_pred=pred_test_y)
    test_mape = mape.result()

    # rmse
    rmse = RootMeanSquaredError()
    rmse.update_state(y_true=valid_y_true, y_pred=pred_valid_y)
    valid_rmse = rmse.result()

    rmse.reset_states()
    rmse.update_state(y_true=test_y_true, y_pred=pred_test_y)
    test_rmse = rmse.result()

    print('valid: mae; {}， mape: {}, rmse: {}'.
          format(valid_mae, valid_mape, valid_rmse))
    print('test: mae; {}， mape: {}, rmse: {}'.
          format(test_mae, test_mape, test_rmse))


def run():
    """

    :return:
    """
    valid_size = 23
    test_size = 23
    file = './data/001632.csv'

    epochs = 100
    batch_size = 200

    df = get_data(file=file)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(X=df)

    valid_begin = -(valid_size + test_size)
    train_data = data[: valid_begin]
    valid_data = data[valid_begin: -test_size]
    test_data = data[-test_size:]

    train_x, train_y, valid_x, valid_y, test_x, test_y = \
        preprocess(train_data=train_data,
                   valid_data=valid_data,
                   test_data=test_data)

    # super parameter grid search
    units_arr = [64, 128, 256, 512, 1024]
    units_arr = [448, 512, 576]
    reg_arr = [0, 0.001, 0.01, 0.1, 0.3]
    activate_arr = ['linear', 'softsign']
    loss_arr = ['mse', 'mae', 'mape']
    opt_arr = ['adam', 'RMSprop', 'adadelta']

    choose_parameter = True
    build_flag = False

    # choose the best parameter
    if choose_parameter:
        for loss in loss_arr:
            print('*' * 100)
            grid_search(train_x, train_y, valid_x, valid_y, test_x, test_y,
                        epochs, batch_size,
                        units_arr, activate_arr=['softsign'],
                        opt_arr=['adam'],
                        loss=['mse'],
                        reg_arr=reg_arr,
                        workers=8)

    # 用best parameter build model
    if build_flag:
        model, loss_metrics = \
            build_fit_model(train_x, train_y,
                            valid_x, valid_y,
                            test_x, test_y,
                            unit=512,
                            act='softsign',
                            loss='mae',
                            opt='RMSprop',
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=1)

        pred_train = model.predict(train_x)
        pred_valid = model.predict(valid_x)
        pred_test = model.predict(test_x)

        # 恢复归一化前
        train_y_true = scaler.inverse_transform(train_data)[:, -1:]
        valid_y_true = scaler.inverse_transform(valid_data)[:, -1:]
        test_y_true = scaler.inverse_transform(test_data)[:, -1:]

        train_y_true = [num[0] for num in train_y_true]
        valid_y_true = [num[0] for num in valid_y_true]
        test_y_true = [num[0] for num in test_y_true]

        # 归一化 恢复
        pred_train_all = train_data.copy()
        pred_train_all[:, -1:] = pred_train
        pred_train_scale = scaler.inverse_transform(pred_train_all)
        pred_train_y = pred_train_scale[:, -1:]
        pred_train_y = [num[0] for num in pred_train_y]

        pred_valid_all = valid_data.copy()
        pred_valid_all[:, -1:] = pred_valid
        pred_valid_scale = scaler.inverse_transform(pred_valid_all)
        pred_valid_y = pred_valid_scale[:, -1:]
        pred_valid_y = [num[0] for num in pred_valid_y]

        pred_test_all = test_data.copy()
        pred_test_all[:, -1:] = pred_test
        pred_test_scale = scaler.inverse_transform(pred_test_all)
        pred_test_y = pred_test_scale[:, -1:]
        pred_test_y = [num[0] for num in pred_test_y]

        # 评估
        evalute(valid_y_true, pred_valid_y, test_y_true, pred_test_y)

        # 训练，验证，测试画图
        plot_y = \
            train_y_true + valid_y_true + test_y_true + df['close'].iloc[
                                                        -1:].tolist()

        plot_pred_train_y = pred_train_y + [None for _ in pred_valid_y] + \
                            [None for _ in pred_test]
        plot_pred_valid_y = [None for _ in train_y_true] + pred_valid_y + \
                            [None for _ in pred_test]
        plot_pred_test_y = [None for _ in train_y_true] + \
                           [None for _ in pred_valid_y] + pred_test_y
        plt.clf()

        plt.plot(df.index, plot_y[: -1], c='b', label='原始数据')
        plt.plot(df.index, plot_pred_train_y, c='r', label='训练预测数据')
        plt.plot(df.index, plot_pred_valid_y, c='g', label='验证预测数据')
        plt.plot(df.index, plot_pred_test_y, c='y', label='测试预测数据')
        plt.legend()

        file = './data/hs300.png'
        plt.savefig(file)
        # plt.show()


if __name__ == '__main__':
    run()
