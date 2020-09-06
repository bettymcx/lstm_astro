"""
LSTM模型预测
"""
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from model.model_evalute import model_evalute
from model.data_preprocess import get_data, preprocess


def predict_run(file):
    """

    :return:
    """
    code = file[::-1]
    code = code[code.index('.') + 1: code.index('/')][::-1]

    valid_size = 23
    test_size = 23
    # file = '../data/001632.csv'

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

    # 模型load
    model = load_model('./data/{}.h5'.format(code))

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
    model_evalute(valid_y_true, pred_valid_y, test_y_true, pred_test_y)

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

    file = './data/{}.png'.format(code)
    plt.savefig(file)
    plt.show()
