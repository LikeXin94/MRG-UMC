import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import random

def get_mask(view_num, data_len, missing_rate):
    # 未考虑类比信息，要求不同类别样本在不同视图中均匀分布
    """Randomly generate incomplete data information, simulate partial view data with complete view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix


def get_mask_my(view_num, Y_list, paired_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask

    """
    k = np.unique(Y_list)
    n = len(Y_list[0])
    # Y_list = Y_list - 1
    matrix = randint(1, 2, size=(n, view_num))  # 假设均为成对样本 matrix 为样本是否存在的指示矩阵
    obv_rate = 1 / view_num
    com_idx = []
    tmp_idx = []
    mis_idx = []
    for v in range(view_num):
        tmp_idx.append([])
        mis_idx.append([])

    # 初始化cell类型数据，保存每类标号. 观测样本# 20220507
    # tmp_idx = np.empty((1, view_num), dtype=object)# 20220507
    # mis_idx = np.empty((1, view_num), dtype=object)# 20220507

    for c in range(1, len(k) + 1):
        # 根据paired_rate 获得成对样本标号
        row = np.array(np.where(Y_list[0] == c))
        tmp_com_idx = row[0][1:int(paired_rate * len(row[0])) + 1:1]  # 存在的成对样本
        com_idx = np.hstack((com_idx, tmp_com_idx))
        # array类型数据转化为list类型
        com_idx = com_idx.tolist()
        # tmp_com_idx = tmp_com_idx.tolist()
        # 得到剩余的idx标号
        row = row.tolist()
        # print('row', row)
        # print('set(row[0]', set(row[0]))
        # print('set(com_idx)', set(com_idx))
        # print('list(set(row[0]).difference(set(com_idx)))', list(set(row[0]).difference(set(com_idx))))
        row = list(set(row[0]).difference(set(com_idx)))# 位置随机
        row = np.array(row)
        row = np.sort(row)# 20230329
        # print('row', row)
        # exit()
        # 对np.array 得到的row 进行排序，与 第74行row比较


        # 对剩余的标号按照视图均分
        for v in range(view_num):
            tmp_idx[v] = np.hstack((tmp_idx[v], row[int(obv_rate * v * len(row)):int(obv_rate * (v + 1) * len(row))]))
            # tmp_idx[v].append(row[int(obv_rate * v * len(row)):int(obv_rate * (v + 1) * len(row))])
        # for v in range(len(k)):
        #     if v % view_num == view_num

        # for v in range(view_num):
        #     tmp_idx[0, v] = np.hstack(
        #         (tmp_idx[0, v], row[int(obv_rate * v * len(row)):int(obv_rate * (v + 1) * len(row))]))

            # tmp_idx = np.hstack((tmp_idx, row[int(obv_rate * v * len(row)):int(obv_rate * (v + 1) * len(row))]))
            # tmp_idx[v] = cat(tmp_idx[v],row(1+int(obv_rate*(v-1)*len(row)),int(obv_rate*v*len(row))),0)

    # 获得缺失样本对应的下标
    # matrix 表示样本缺失的指示矩阵
    row = np.array(range(n))
    row = row.tolist()
    for v in range(view_num):
        tmp_idx[v] = np.sort(tmp_idx[v])
        # mis_idx[v] = list(set(row).difference(set(tmp_idx[v])))
        mis_idx[v] = np.sort(list(set(row).difference(set(tmp_idx[v]))))# 20230330
        matrix[:, v][mis_idx[v]] = 0

    # for v in range(view_num):
    #     mis_idx[0, v] = list(set(row).difference(set(tmp_idx[0, v])))
    #     matrix[:, v][mis_idx[0, v]] = 0
    return matrix, tmp_idx, mis_idx




def get_mask_shuffle_my(view_num, x_train_raw, Y_list, paired_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask

    """
    k = np.unique(Y_list)
    n = len(Y_list[0])

    # shuffle
    a = list(range(len(Y_list[0])))
    np.random.seed(1)
    random.shuffle(a)
    x_train_raw1 = x_train_raw.copy()
    Y_label = Y_list.copy()
    for v in range(view_num):
        x_train_raw1[v] = x_train_raw1[v][a]
    Y_label[0] = Y_label[0][a]

    # Y_list = Y_list - 1
    matrix = randint(1, 2, size=(n, view_num))  # 假设均为成对样本 matrix 为样本是否存在的指示矩阵
    obv_rate = 1 / view_num
    com_idx = []
    tmp_idx = []
    mis_idx = []
    for v in range(view_num):
        tmp_idx.append([])
        mis_idx.append([])

    for c in range(1, len(k) + 1):
        # 根据paired_rate 获得成对样本标号
        row = np.array(np.where(Y_label[0] == c))
        tmp_com_idx = row[0][1:int(paired_rate * len(row[0])) + 1:1]  # 存在的成对样本
        com_idx = np.hstack((com_idx, tmp_com_idx))
        # array类型数据转化为list类型
        com_idx = com_idx.tolist()
         # 得到剩余的idx标号
        row = row.tolist()

        row = list(set(row[0]).difference(set(com_idx)))# 位置随机
        row = np.array(row)
        row = np.sort(row)# 20230329


        # 对剩余的标号按照视图均分
        for v in range(view_num):
            tmp_idx[v] = np.hstack((tmp_idx[v], row[int(obv_rate * v * len(row)):int(obv_rate * (v + 1) * len(row))]))# 1000*1
            # tmp_idx[v].append(row[int(obv_rate * v * len(row)):int(obv_rate * (v + 1) * len(row))])


        # for v in range(view_num):
        #     tmp_idx[0, v] = np.hstack(
        #         (tmp_idx[0, v], row[int(obv_rate * v * len(row)):int(obv_rate * (v + 1) * len(row))]))


    # 获得缺失样本对应的下标
    # matrix 表示样本缺失的指示矩阵
    row = np.array(range(n))
    row = row.tolist()
    tmp_idx1 = tmp_idx.copy()
    for v in range(view_num):
        # tmp_idx1[v] = np.sort(tmp_idx1[v])
        mis_idx[v] = list(set(row).difference(set(tmp_idx1[v])))
        # mis_idx[v] = np.sort(list(set(row).difference(set(np.sort(tmp_idx[v])))))# 20230330
        matrix[:, v][mis_idx[v]] = 0

    # shuffle 后，不同类别数据按照类别顺序排列
    x_train_raw2 = []
    for v in range(view_num):
        x_train_raw2.append(x_train_raw1[v](tmp_idx[v]))  # 1000*d
    Y_label[0] = Y_label[0](tmp_idx[0].extend(tmp_idx[1]))
    x_train_raw = x_train_raw1.copy()

    # for v in range(view_num):
    #     mis_idx[0, v] = list(set(row).difference(set(tmp_idx[0, v])))
    #     matrix[:, v][mis_idx[0, v]] = 0
    return x_train_raw2, Y_label, matrix, tmp_idx, mis_idx





