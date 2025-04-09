import argparse
import collections
import itertools
import torch
from model_MRG_UMC_autopara_tuning_6view import *  
from get_mask import get_mask_my
from util import cal_std, get_logger, cal_std_my
from datasets import *
from configure import get_default_config
from sklearn.preprocessing import StandardScaler  

import sys
stdout_backup = sys.stdout
log_file = open("message.log", "w")
sys.stdout = log_file


# dataset
dataset = {
    1: 'Caltech101-20-6view',
    2: "digit_6view",# 6view#[76,216,64,240,47,6][2000]
}
# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='2', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='40', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='1', help='number of test times')
parser.add_argument('--tau', type=float, default='0.1', help='hyperparameter in contrastive learning')
args = parser.parse_args()

dataset = dataset[args.dataset]

def main():
    use_cuda = torch.cuda.is_available() #检验当前GPU是否可用
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)#
    config['print_num'] = args.print_num# 100
    config['dataset'] = dataset
    config['test_time'] = args.test_time
    logger = get_logger()


    # Load data
    # complete sample
    X_list, Y_list = load_data_my(config)
    x_train_raw1 = X_list.copy()
    view_num = len(X_list)
    Y_label = Y_list.copy()




    accumulated_metrics_lambda1 = collections.defaultdict(list)
    best_result_metrics = collections.defaultdict(list)  # 为字典提供默认值，
    paired_rate = 0

    "超参数调参过程，lambda1, lambda2, lambda3, lambda4, 为本文超参数"


    for hyper_lambda1 in range(len(config['training']['lambda1'])):# alignment_loss
        for hyper_lambda2 in range(len(config['training']['lambda2'])):# inter_contrastive
            for hyper_lambda3 in range(len(config['training']['lambda3'])):# cross_contrastive
                for hyper_lambda4 in range(len(config['training']['lambda4'])):  # L_z_norm
                    args.test_time = 1
                    for data_seed in range(1, args.test_time + 1):
                        # Get the Mask; 得到缺失样本对应的下标
                        np.random.seed(data_seed)

                        # mask = get_mask(2, x1_train_raw.shape[0], config['training']['missing_rate'])
                        mask, tmp_idx, mis_idx = get_mask_my(view_num, Y_label, paired_rate)
                        mask = torch.from_numpy(mask).long().to(device)

                        # 对于数据集本身类别对应样本数目不一样高，从导致构造不同视图构造样本数目不一致的问题，做一些微调
                        # [187, 187, 204, 187, 204, 187, 204, 187]，均取最小值
                        a = min(len(tmp_idx[0]), len(tmp_idx[1]), len(tmp_idx[2]), len(tmp_idx[3]),
                                                len(tmp_idx[4]), len(tmp_idx[5]), )
                        for v in range(view_num):
                            if len(tmp_idx[v]) > a:
                                tmp_idx[v] = tmp_idx[v][0:a]

                        x_train_raw = []
                        Y_view = []
                        for v in range(0, view_num):
                            a = np.array(tmp_idx[v]).astype(int)
                            x_train_raw.append(x_train_raw1[v][a])
                            Y_view.append(Y_label[0][a])
                            Y_label_adjust = np.concatenate([Y_view[0], Y_view[1], Y_view[2], Y_view[3],
                                                                             Y_view[4], Y_view[5]], axis=0)

                        # Set random seeds
                        if config['training']['missing_rate'] == 0:
                            seed = data_seed
                        else:
                            seed = config['training']['seed']
                        np.random.seed(seed)
                        random.seed(seed + 1)
                        torch.manual_seed(seed + 2)
                        torch.cuda.manual_seed(seed + 3)
                        torch.backends.cudnn.deterministic = True

                        COMPLETER_my = Completer_my_20220605(config)

                        optimizer = torch.optim.Adam(
                                                itertools.chain(COMPLETER_my.autoencoder1.parameters(), COMPLETER_my.autoencoder2.parameters(),
                                                                COMPLETER_my.autoencoder3.parameters(), COMPLETER_my.autoencoder4.parameters(),
                                                                COMPLETER_my.autoencoder5.parameters(), COMPLETER_my.autoencoder6.parameters(),
                                                                COMPLETER_my.img2txt.parameters(), COMPLETER_my.txt2img.parameters()),
                                                lr=config['training']['lr'])
                        COMPLETER_my.to_device(device)

                        # Print the models
                        logger.info(COMPLETER_my.autoencoder1)
                        logger.info(COMPLETER_my.img2txt)
                        logger.info(optimizer)

                        print('oktoday')

                        standardscaler = StandardScaler()
                        # 对数组x遍历，对每一个样本进行标准化
                        x_train = []
                        for v in range(view_num):
                            scaler = standardscaler.fit(x_train_raw[v])  # 存储计算得到的均值和方差
                            x_train.append(scaler.transform(x_train_raw[v]))  # 返回标准化后的样本集


                        acc, nmi, pre, f_measure, recall, ari, ami, acc_show, nmi_show, precision_show, F_measure_show, recall_show, ARI_show, AMI_show, d_A_distance_record \
                                                = COMPLETER_my.train(config, logger, x_train, Y_label_adjust, mask, optimizer, device,
                                                                     tmp_idx, hyper_lambda1, hyper_lambda2, hyper_lambda3,
                                                                     hyper_lambda4)
                        accumulated_metrics_lambda1['acc'].append(acc)
                        accumulated_metrics_lambda1['nmi'].append(nmi)
                        accumulated_metrics_lambda1['Precision'].append(pre)
                        accumulated_metrics_lambda1['F_score'].append(f_measure)
                        accumulated_metrics_lambda1['Recall'].append(recall)
                        accumulated_metrics_lambda1['ari'].append(ari)
                        accumulated_metrics_lambda1['AMI'].append(ami)


                    logger.info('--------------------Training over--------------------')
                    cal_std_my(logger, accumulated_metrics_lambda1['acc'], accumulated_metrics_lambda1['nmi'], accumulated_metrics_lambda1['Precision'],
                                                   accumulated_metrics_lambda1['F_score'], accumulated_metrics_lambda1['Recall'],accumulated_metrics_lambda1['ari'],
                                                   accumulated_metrics_lambda1['AMI'],)
                    sio.savemat('accumulated.mat', {'accumulated_metrics_lambda1': accumulated_metrics_lambda1})

                    t_show = np.sum([acc_show, nmi_show, F_measure_show], axis=0)
                    t = np.where(t_show == np.max(t_show))
                    best_result_idx = list(t)[0][0]

                    best_result_metrics['resul_idx'].append(best_result_idx)
                    best_result_metrics['acc'].append(acc_show[best_result_idx])
                    best_result_metrics['nmi'].append(nmi_show[best_result_idx])
                    best_result_metrics['Precision'].append(precision_show[best_result_idx])
                    best_result_metrics['F_score'].append(F_measure_show[best_result_idx])
                    best_result_metrics['recall'].append(recall_show[best_result_idx])
                    best_result_metrics['ARI'].append(ARI_show[best_result_idx])
                    best_result_metrics['AMI'].append(AMI_show[best_result_idx])
                    best_result_metrics1 = best_result_metrics
                    print('best_result_metrics1', best_result_metrics1)
                    sio.savemat('best_result_metrics1.mat', {'best_result_metrics1': best_result_metrics1})
                    # sio.savemat('best_result_metrics.mat', {'best_result_metrics': best_result_metrics})
                    print('d_A_distance_record', d_A_distance_record)

        # message.log文件关闭
    log_file.close()
    sys.stdout = stdout_backup
if __name__ == '__main__':
    main()
