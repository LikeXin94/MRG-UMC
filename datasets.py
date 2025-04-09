import os, random, sys
import numpy as np
import scipy.io as sio
import util
from sklearn.preprocessing import MinMaxScaler# 对数据标准化处理
tool = MinMaxScaler(feature_range=(0, 1))



def load_data_my(config):
    """Load data """
    data_name = config['dataset']
    # main_dir = sys.path[0]
    X_list = []
    Y_list = []

    main_dir = os.path.dirname(os.getcwd()) # 上层目录

    if data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '.mat'))
        X = mat['X'][0]
        for view in [3, 4]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)


    elif data_name in ['Caltech101-20-6view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'Caltech101-20.mat'))
        X = mat['X'][0]
        for view in range(6):
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['digit_2view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'digit.mat'))
        X = mat['tmpX'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list[0] = tool.fit_transform(X_list[0])
        X_list[1] = tool.fit_transform(X_list[1])
        Y_list.append(np.squeeze(mat['tmpY']))



    elif data_name in ['digit_6view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'digit_6views.mat'))
        X = mat['tmpX'][0]
        for v in range(6):
            X_list.append(X[v].astype('float32'))
            X_list[v] = tool.fit_transform(X_list[v])
        Y_list.append(np.squeeze(mat['tmpY']))



    return X_list, Y_list

