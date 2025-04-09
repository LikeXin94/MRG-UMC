import logging
import numpy as np
import math

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def next_batch(X1, X2, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, (i + 1))


def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        logger.info('ACC:'+ str(arg[0]))
        logger.info('NMI:'+ str(arg[1]))
        logger.info('ARI:'+ str(arg[2]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
                                                                                                 np.std(arg[0]) * 100,
                                                                                                 np.mean(arg[1]) * 100,
                                                                                                 np.std(arg[1]) * 100,
                                                                                                 np.mean(arg[2]) * 100,
                                                                                                 np.std(arg[2]) * 100)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
    logger.info(output)

    return

def cal_std_my(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        logger.info('ACC:'+ str(arg[0]))
        logger.info('NMI:'+ str(arg[1]))
        logger.info('ARI:'+ str(arg[2]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
                                                                                                 np.std(arg[0]) * 100,
                                                                                                 np.mean(arg[1]) * 100,
                                                                                                 np.std(arg[1]) * 100,
                                                                                                 np.mean(arg[2]) * 100,
                                                                                                 np.std(arg[2]) * 100)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
    elif len(arg) == 7:
        logger.info('ACC:' + str(arg[0]))
        logger.info('NMI:' + str(arg[1]))
        logger.info('Precision:' + str(arg[2]))
        logger.info('F_measure:' + str(arg[3]))
        logger.info('recall:' + str(arg[4]))
        logger.info('ARI:' + str(arg[5]))
        logger.info('AMI:' + str(arg[6]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} Precision {:.2f} std {:.2f}  F_measure {:.2f} std {:.2f}
         recall {:.2f} std {:.2f} ARI {:.2f} std {:.2f} AMI {:.2f} std {:.2f}""".format(
            np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
            np.mean(arg[1]) * 100, np.std(arg[1]) * 100,
            np.mean(arg[2]) * 100, np.std(arg[2]) * 100,
            np.mean(arg[3]) * 100, np.std(arg[3]) * 100,
            np.mean(arg[4]) * 100, np.std(arg[4]) * 100,
            np.mean(arg[5]) * 100, np.std(arg[5]) * 100,
            np.mean(arg[6]) * 100, np.std(arg[5]) * 100,
        )
    logger.info(output)
    return


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
def next_batch_multiview_flower17_3views(X1, X2, X3, batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908

        yield (batch_x1, batch_x2, batch_x3, (i + 1))# 20220908

def next_batch_multiview_digit(X1, X2, X3, X4, X5, batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, (i + 1))# 20220908

def next_batch_multiview_digit_6views(X1, X2, X3, X4, X5, X6, batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]
        batch_x6 = X6[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, (i + 1))# 20220908


def next_batch_multiview_flower17_7views(X1, X2, X3, X4, X5, X6, X7,batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]
        batch_x6 = X6[start_idx: end_idx, ...]
        batch_x7 = X7[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_x7,(i + 1))# 20220908