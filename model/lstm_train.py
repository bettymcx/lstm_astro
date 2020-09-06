"""
LSTM模型训练
"""
from sklearn.preprocessing import MinMaxScaler
from model.model_evalute import build_fit_model
from model.data_preprocess import get_data, preprocess


def train_run(file, unit, act, loss, opt, l2_ratio):
    """

    :param file:
    :param unit:
    :param act:
    :param loss:
    :param opt:
    :param l2_ratio:
    :return:
    """
    code = file[::-1]
    code = code[code.index('.') + 1: code.index('/')][::-1]

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

    # 用best parameter build model
    model, _ = \
        build_fit_model(train_x, train_y,
                        valid_x, valid_y,
                        test_x, test_y,
                        unit=unit,
                        act=act,
                        loss=loss,
                        opt=opt,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        l2_ratio=l2_ratio)

    # 模型save
    model.save('./data/{}.h5'.format(code))
