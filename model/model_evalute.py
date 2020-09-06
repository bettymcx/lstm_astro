"""
LSTM模型build，fit & evaluite
"""
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import L2
from keras.metrics import (RootMeanSquaredError,
                           MeanAbsoluteError,
                           MeanAbsolutePercentageError)
from model.plot_graph import plot_info


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
                  ylabel='Loss')

        # acc
        if 'acc' in history.history.keys():
            train_acc = history.history['acc']
            test_val_acc = history.history['val_acc']
            plot_info(train=train_acc, test=test_val_acc,
                      title='LSTM 模型准确率',
                      ylabel='Accuracy')
    return model, loss_metrics


def model_evalute(valid_y_true, pred_valid_y, test_y_true, pred_test_y):
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
