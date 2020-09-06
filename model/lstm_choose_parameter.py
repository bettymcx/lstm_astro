"""
LSTM模型参数择优
"""
from sklearn.preprocessing import MinMaxScaler
from model.model_evalute import build_fit_model
from model.data_preprocess import get_data, preprocess


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


def choose_parameter_run(file):
    """

    :param file:
    :return:
    """
    valid_size = 23
    test_size = 23
    # file = '../data/001632.csv'

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
    # units_arr = [400, 448, 512, 576]
    units_arr = [16, 32, 64, 96]
    reg_arr = [0, 0.001, 0.01, 0.1, 0.3]
    activate_arr = ['linear', 'softsign', 'relu']
    loss_arr = ['mse', 'mae', 'mape']
    opt_arr = ['adam', 'RMSprop', 'adadelta']

    for loss in loss_arr:
        print('*' * 100)
        grid_search(train_x, train_y, valid_x, valid_y, test_x, test_y,
                    epochs, batch_size,
                    units_arr, activate_arr=['softsign'],
                    opt_arr=['RMSprop'],
                    loss=loss,
                    reg_arr=reg_arr,
                    workers=8)
