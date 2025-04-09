import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np
from sklearn.cluster import KMeans
from evaluation import clustering_metric
import evaluation
from util import next_batch_multiview_flower17_7views, next_batch_multiview_digit_6views  # next_batch,
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import scipy.io as sio
import random
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import linalg
import math



class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class Clustering(nn.Module):
    def __init__(self, K, d):
        super(Clustering, self).__init__()
        # input_A = 784  input_B = 784
        # self.commonz = input1
        self.weights = nn.Parameter(torch.randn(K, d).cuda(), requires_grad=True)

    #        self.layer1 = nn.Linear(d, K, bias = False)

    def forward(self, comz):
        q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(comz, 1) - self.weights, 2), 2)))
        q = torch.t(torch.t(q1) / torch.sum(q1))
        loss_q = torch.log(q)
        return loss_q, q


class Completer_my_20220605():
    """COMPLETER module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config
        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')  # check the last dimension

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']  # !!!!!!!!!!!20220907再检查,不需要

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder3 = Autoencoder(config['Autoencoder']['arch3'], config['Autoencoder']['activations3'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder4 = Autoencoder(config['Autoencoder']['arch4'], config['Autoencoder']['activations4'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder5 = Autoencoder(config['Autoencoder']['arch5'], config['Autoencoder']['activations5'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder6 = Autoencoder(config['Autoencoder']['arch6'], config['Autoencoder']['activations6'],
                                        config['Autoencoder']['batchnorm'])

        # Dual predictions.
        # To illustrate easily, we use "img" and "txt" to denote two different views.
        self.img2txt = Prediction(self._dims_view1)  # 20220907再检查，不需要
        self.txt2img = Prediction(self._dims_view2)



    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.autoencoder3.to(device)
        self.autoencoder4.to(device)
        self.autoencoder5.to(device)
        self.autoencoder6.to(device)

        self.img2txt.to(device)
        self.txt2img.to(device)


    def show_Training_history_loss(self, loss_show, loss_show_rec, loss_show_z_norm,
                                       loss_show_inner_contrastive, loss_show_cross_contrastive, loss_show_KL, test_time):
        # 训练数据执行结果，’-‘表示实线，’b'表示蓝色
        plt.figure()
        x = [i for i in range(len(loss_show))]
        plt.plot(x, loss_show, linestyle='-', color='r')
        plt.plot(x, loss_show_z_norm, linestyle='-.', color='g')
        plt.plot(x, loss_show_rec, linestyle='-.', color='gold')
        plt.plot(x, loss_show_inner_contrastive, linestyle='-.', color='b')
        plt.plot(x, loss_show_cross_contrastive, linestyle='-.', color='magenta')
        plt.plot(x, loss_show_KL, linestyle='-.', color='cyan')

        # 显示图的标题
        plt.title('Training loss history')
        # 显示x轴标签epoch
        plt.xlabel('epoch')
        # 显示y轴标签train
        plt.ylabel('train_loss')
        plt.legend(['training_loss', 'z_norm_loss', 'rec_loss', 'inner_contrastive_loss', 'cross_contrastive_loss', 'KL_loss'], loc='upper right')
        fig = plt.gcf()  # 获取当前图像
        fig.savefig(
            r'/home/learning_code/test_autoencoder_20230312/figs/my_loss_term{}.png'.format(test_time))  # 127
        # 开始绘图
        plt.show()
        fig.clear()  # 释放内存

    def show_Training_history(self, loss_show, test_time):
        # 训练数据执行结果，’-‘表示实线，’b'表示蓝色
        x = [i for i in range(len(loss_show))]
        plt.plot(x, loss_show, linestyle='-', color='r')
        # 显示图的标题
        plt.title('Training loss history')
        # 显示x轴标签epoch
        plt.xlabel('epoch')
        # 显示y轴标签train
        plt.ylabel('train_loss')
        # 设置图例是显示'train','validation',位置在右下角
        plt.legend(['training_loss'], loc='upper right')
        fig = plt.gcf()  # 获取当前图像
        fig.savefig(r'/home/learning_code/test_autoencoder_20230312/figs/super_class/caltech101_my_loss_231230_{}.eps'.format(test_time))  # 127
        # 开始绘图
        plt.show()
        fig.clear()  # 释放内存


    def show_acc_nmi_history(self, acc_show, nmi_show, precision_show, F_measure_show, test_time):
        # 训练数据执行结果，’-‘表示实线，’b'表示蓝色
        x = [i for i in range(len(acc_show))]
        plt.plot(x, acc_show, linestyle='-', color='r')
        plt.plot(x, nmi_show, linestyle='-', color='g')
        plt.plot(x, precision_show, linestyle='-', color='b')
        plt.plot(x, F_measure_show, linestyle='-', color='gold')
        # # 验证数据执行结果，‘--’表示虚线，‘r'表示红色
        # plt.plot(Training.history[validation], linestyle='--', color='r')
        # 显示图的标题
        plt.title('Training acc_nmi_history')
        # 显示x轴标签epoch
        plt.xlabel('epoch')
        # 显示y轴标签train
        plt.ylabel('acc_nmi')
        # 设置图例是显示'train','validation',位置在右下角
        plt.legend(['acc', 'nmi', 'precision', 'F_measure'], loc='lower right')
        fig = plt.gcf()  # 获取当前图像
        fig.savefig(
            r'/home/learning_code/test_autoencoder_20230312/figs/super_class/digit_my_acc_nmi_231230_{}.eps'.format(test_time))  # 127
        # 开始绘图
        plt.show()
        fig.clear()  # 释放内存

    def plot_embedding(self, data, label, title):
        """
        :param data:数据集
        :param label:样本标签
        :param title:图像标题
        :return:图像
        """
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
        fig = plt.figure()  # 创建图形实例
        ax = plt.subplot(111)  # 创建子图
        # 遍历所有样本
        k = len(np.unique(label))
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / k),
                     fontdict={'weight': 'bold', 'size': 7})
        plt.xticks()  # 指定坐标的刻度
        plt.yticks()
        plt.title(title, fontsize=14)
        # 返回值
        return fig

    def plot_embedding_zviews(self, sample_view1, sample_view2, label_view1, label_view2, title1, title2):
        x_min, x_max = np.min(sample_view1, 0), np.max(sample_view1, 0)
        data = (sample_view1 - x_min) / (x_max - x_min)  # 对数据进行归一化处理
        fig = plt.figure()  # 创建图形实例
        ax = plt.subplot(121)  # 创建子图
        # 遍历所有样本
        k = len(np.unique(label_view1))
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], str(label_view1[i]), color=plt.cm.Set1(label_view1[i] / k),
                     fontdict={'weight': 'bold', 'size': 7})
        plt.xticks()  # 指定坐标的刻度
        plt.yticks()
        plt.title(title1, fontsize=12)

        x_min, x_max = np.min(sample_view2, 0), np.max(sample_view2, 0)
        data = (sample_view2 - x_min) / (x_max - x_min)  # 对数据进行归一化处理

        # fig = plt.figure()  # 创建图形实例
        bx = plt.subplot(122)  # 创建子图
        # 遍历所有样本
        k = len(np.unique(label_view2))
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], str(label_view2[i]), color=plt.cm.Set1(label_view2[i] / k),
                     fontdict={'weight': 'bold', 'size': 7})
        plt.xticks()  # 指定坐标的刻度
        plt.yticks()
        plt.title(title2, fontsize=12)
        # 返回值
        return fig

    def test_tsne_my(self, data, label, index):
        ts = TSNE(n_components=2, init='pca', random_state=0)
        # t-SNE降维
        result = ts.fit_transform(data)
        # 调用函数，绘制图像
        fig = self.plot_embedding(result, label, 't-SNE Embedding of digit', )

        fig = plt.gcf()  # 获取当前图像
        # fig.savefig(r'D:\{}.png'.format())
        fig.savefig(r'/home/learning_code/test_autoencoder_20230312/figs/super_class/{}.eps'.format(index))  # 127

        plt.show()
        fig.clear()  # 释放内存

    # label vector converted to label matrix
    def label_to_matrix(self, label):
        label = np.array(label)
        uq_la = np.unique(label)
        c = uq_la.shape[0]
        n = len(label)
        # n = label.shape[0]
        label_mat = np.zeros((n, c))
        for i in range(c):
            index = (label == i + 1)
            label_mat[index, i] = 1.0
        return label_mat



    def train(self, config, logger, x_train, Y_list, mask, optimizer, device, tmp_idx, hyper_lambda1, hyper_lambda2,
                  hyper_lambda3, hyper_lambda4):

        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari
        """
        view_num = len(x_train)
        # print(view_num)
        for v in range(view_num):
            x_train[v] = torch.as_tensor(x_train[v]).to(device)
        train_view = x_train.copy()

        # Get the unpaired data for training
        n_clusters = np.unique(Y_list)
        K = len(n_clusters)
        d_subspace = 128  # 128#512#256#

        # 单视图 label
        Y_lable_single, a = [], []
        b = 0
        for v in range(view_num):
            a.append(len(train_view[v]))
            Y_lable_single.append(Y_list[b:sum(a)])
            b = sum(a)

        # training
        '''show loss and show measure'''
        loss_show, loss_show_silhouette, loss_show_clustering, loss_show_rec1, loss_show_rec2 = [], [], [], [], []

        acc_show, nmi_show, precision_show, F_measure_show = [], [], [], []
        recall_show, ARI_show, AMI_show = [], [], []
        result_single_view_all = []
        iter_current_epoch = 0
        d_A_distance_record = []
        '''针对K的取值'''
        curr_K = [2, math.ceil(K // 2), K]

        # 每10个epoch为一个单位，计算当前K下的超类效果
        inter_val = 25  # 20  # 30  # 20#10
        sil_para = 1.5
        step_gamma = 0.99


        for epoch in range(config['training']['epoch']):
            it = int(iter_current_epoch / inter_val)
            if iter_current_epoch + 1 == config['training']['epoch'][0] or it >= len(curr_K):
                it = len(curr_K) - 1

            shuffle_idx = np.arange(len(train_view[0]))
            X1, X2, X3, X4, X5, X6, shuffle_idx = shuffle(train_view[0], train_view[1], train_view[2], train_view[3],
                                                          train_view[4], train_view[5], shuffle_idx)
            '''中间值'''
            loss_all, loss_rec1, loss_rec2, loss_z_norm1, loss_structure_inner_contrastive1, loss_structure_KL1, loss_rec12 \
                = 0, 0, 0, 0, 0, 0, 0
            loss_structure_cross_contrastive1 = 0

            H = []
            for v in range(view_num):
                H.append(np.random.random(size=(d_subspace, d_subspace)))
                H[v] = H[v].astype(np.float32)
            H = torch.as_tensor(H).to(device)

            for batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_No in next_batch_multiview_digit_6views(
                    X1,
                    X2, X3, X4, X5, X6, config['training']['batch_size']):

                structure_loss_KL, structure_loss_inner_contrastive, structure_loss_cross_contrastive = 0, 0, 0

                z_before = []
                batch_x1 = batch_x1.cuda()
                z_before.append(self.autoencoder1.encoder(batch_x1))
                z_before.append(self.autoencoder2.encoder(batch_x2))
                z_before.append(self.autoencoder3.encoder(batch_x3))
                z_before.append(self.autoencoder4.encoder(batch_x4))
                z_before.append(self.autoencoder5.encoder(batch_x5))
                z_before.append(self.autoencoder6.encoder(batch_x6))

                z = []
                commonZ_term = []
                for v in range(view_num):
                    z.append(torch.as_tensor(z_before[v]).clone())
                    commonZ_term.append(torch.as_tensor(z_before[v]).clone().tolist)

                commonZ_term_before = []
                for v in range(view_num):
                    commonZ_term_before.append(torch.as_tensor(z_before[v]).clone())

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_before[0]), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_before[1]), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_before[2]), batch_x3)
                recon4 = F.mse_loss(self.autoencoder4.decoder(z_before[3]), batch_x4)
                recon5 = F.mse_loss(self.autoencoder5.decoder(z_before[4]), batch_x5)
                recon6 = F.mse_loss(self.autoencoder6.decoder(z_before[5]), batch_x6)

                reconstruction_loss = recon1 + recon2 + recon3 + recon4 + recon5 + recon6

                '''1 正交约束'''
                loss_z_norm = 0
                for v in range(view_num):
                    loss_z_norm += torch.norm(
                        z_before[v].t() @ z_before[v] - torch.as_tensor(np.eye(d_subspace).astype(np.float32)).to(
                            device), p=2)

                '''20230927 super-class'''
                '''目标： 关于对比学习的构造，小K的结果指引大K的结果'''
                # K=10时, 聚类结果，用于指引正负样本对的构建
                label_views_K1, label_views_K1_matrix = [], []
                current_idx_K1_view = []  # 不同视图、不同类别下标索引： [v1[c1], v1[c2],... ], [v2[c1], v2[c2],...]
                centroids_views = []
                for v in range(view_num):
                    estimator = KMeans(curr_K[0]).fit(commonZ_term_before[v].cpu().detach().numpy())
                    centroids_views.append(estimator.cluster_centers_)
                    label_pred = estimator.labels_
                    label_views_K1.append([label_pred])
                    label_views_K1_matrix.append(self.label_to_matrix(label_pred).astype(np.float32))
                    '''record the label_pred idx in K=10 in one view'''
                    current_idx_K1 = []
                    for i in range(curr_K[0]):
                        current_idx_K1.append(np.array(np.where(label_pred == i)))
                    current_idx_K1_view.append(current_idx_K1)

                label_views_K2, label_views_K2_matrix = [], []
                current_idx_K2_view = []  # 不同视图、不同类别下标索引： [v1[c1], v1[c2],... ], [v2[c1], v2[c2],...]
                centroids_views = []
                for v in range(view_num):
                    estimator = KMeans(curr_K[1]).fit(commonZ_term_before[v].cpu().detach().numpy())
                    centroids_views.append(estimator.cluster_centers_)
                    label_pred = estimator.labels_
                    label_views_K2.append([label_pred])
                    label_views_K2_matrix.append(self.label_to_matrix(label_pred).astype(np.float32))
                    '''record the label_pred idx in K=10 in one view'''
                    current_idx_K2 = []
                    for i in range(curr_K[1]):
                        current_idx_K2.append(np.array(np.where(label_pred == i)))
                    current_idx_K2_view.append(current_idx_K2)

                # label_views_K3, label_views_K3_matrix = [], []
                # current_idx_K3_view = []  # 不同视图、不同类别下标索引： [v1[c1], v1[c2],... ], [v2[c1], v2[c2],...]
                # centroids_views = []
                # for v in range(view_num):
                #     estimator = KMeans(curr_K[2]).fit(commonZ_term_before[v].cpu().detach().numpy())
                #     centroids_views.append(estimator.cluster_centers_)
                #     label_pred = estimator.labels_
                #     label_views_K3.append([label_pred])
                #     label_views_K3_matrix.append(self.label_to_matrix(label_pred).astype(np.float32))
                #     '''record the label_pred idx in K=10 in one view'''
                #     current_idx_K3 = []
                #     for i in range(curr_K[2]):
                #         current_idx_K3.append(np.array(np.where(label_pred == i)))
                #     current_idx_K3_view.append(current_idx_K3)

                # 计算不同视图的轮廓系数，轮廓系数大的视图为logits_x_ulb_w
                # 轮廓系数大的视图为logits_x_ulb_s
                silhouette_score_avg, centroids_views, label_views, label_views_matrix = [], [], [], []

                '''2 多层视图内对比学习，利用超类构造正负样本对'''
                # 真正样本对： 在K=k 和K=10时均为同一类时为正对；
                # 真负样本对： 在K=k 和K=10时均不为同一类，则为负对；
                # 假正样本对：在K=k是为同类，在k=10时不为同类，为假正对 （10类时的边界样本，10类时的不同类样本）【可能不存在】
                # 假负样本对：在K=k是为异类，在k=10时为同类，则为假负对 （2类时的边界样本）
                '''contrastive loss - 1：inter_view contrastive'''
                # 记录同类别样本,current_idx 是以当前视图中样本的总数为idx;


                current_idx_k_view = []
                for v in range(view_num):
                    # 当前K = k 时，聚类分配，轮廓系数
                    estimator = KMeans(curr_K[it]).fit(commonZ_term_before[v].cpu().detach().numpy())
                    centroids_views.append(estimator.cluster_centers_)
                    label_pred = estimator.labels_
                    label_views.append([label_pred])
                    '''record the label_pred idx in K=2 in one view'''
                    current_idx_k = []
                    for i in range(curr_K[it]):  # 2类
                        current_idx_k.append(np.array(np.where(label_pred == i)))
                    current_idx_k_view.append(current_idx_k)  # 多个视图，多个类的下标索引

                    # label_views.append(np.diag(label_pred))
                    '''view's silhouette_score'''
                    # 当前类下（2类）的预测结果，对应的轮廓系数值
                    silhouette_score_avg1 = silhouette_score(commonZ_term_before[v].cpu().detach().numpy(), label_pred)
                    silhouette_score_avg.append(silhouette_score_avg1)

                    h = torch.nn.functional.normalize(z[v], p=2, dim=1)
                    similarities = h @ h.t() / config['training']['tau_inter']

                    # similarities 为单个视图的 相似性矩阵，这里以类别为区分（同时考虑10分类时类别关系），得到正样本对
                    nv = len(similarities)

                    loss_inter_contrastive = 0 #同一视图的所有样本
                    # it = 1
                    for i in range(nv):  # 当前视图样本的长度
                        # # 当前样本的类别;digit:[0,1]
                        c = label_pred[i]
                        # 与样本i有相同类别的样本的 idx：  current_idx[c]
                        sample_idx_in_k = current_idx_k[c]
                        if it == 0:
                            pos_idx = sample_idx_in_k[0]
                            neg_idx = np.setdiff1d(np.array(range(nv)), sample_idx_in_k)
                        elif it == 1:  # K=2指引K=5
                            # 检查是否在其他 较小 K分类中也有相同的类别
                            c_K = label_views_K1[v][0][i]  # 在K（2）中的类别标签
                            sample_idx_in_K1 = current_idx_K1_view[v][c_K]  # 在K（2）中的类别标签
                            # 两者之间的交集为pos,既在2类中，也在10类中
                            pos_idx = np.intersect1d(sample_idx_in_k,
                                                     sample_idx_in_K1)  # sample_idx_in_k 当前k， sample_idx_in_K 之前K
                            # 两者之间的合集的补集为 neg
                            neg_idx = np.setdiff1d(np.array(range(nv)), np.union1d(sample_idx_in_k, sample_idx_in_K1))


                        elif it == 2:  # K=2，K=5 指引K=10
                            c_K = label_views_K1[v][0][i]  # 在K（2）中的类别标签
                            sample_idx_in_K1 = current_idx_K1_view[v][c_K]  # 在K（=2）中的类别标签
                            c_K = label_views_K2[v][0][i]  # 在K（2）中的类别标签
                            sample_idx_in_K2 = current_idx_K2_view[v][c_K]  # 在K（=10）中的类别标签
                            # 两者之间的交集为pos,既在2类中，也在10类中
                            pos_idx = np.intersect1d(np.intersect1d(sample_idx_in_k, sample_idx_in_K1),
                                                     sample_idx_in_K2)
                            # 两者之间的合集的补集为 neg
                            neg_idx = np.setdiff1d(np.array(range(nv)),
                                                   np.union1d(np.union1d(sample_idx_in_k, sample_idx_in_K1),
                                                              sample_idx_in_K2))
                        if len(pos_idx) == 0 or len(neg_idx) == 0:
                            continue

                        pos_value = torch.sum(torch.exp(similarities[i, [pos_idx]]), dim=1)  # 20220617
                        neg_value = torch.sum(torch.exp(similarities[i, [neg_idx]]), dim=1)  # 20220613
                        loss_inter_contrastive += sum(- torch.log(pos_value / neg_value))  #第v个视图所有样本

                    structure_loss_inner_contrastive += loss_inter_contrastive /(nv * view_num)# 所有视图的所有样本



                '''contrastive loss - 3: cross_view contrastive with Z^*, i.e., synthesized-view alignment'''
                # 当前current_K下的聚类结果
                latent_fusion_z_common = torch.cat([z[0], z[1], z[2], z[3], z[4], z[5]], dim=0)
                estimator = KMeans(curr_K[it]).fit(latent_fusion_z_common.cpu().detach().numpy())
                centroids_views_zcommon = estimator.cluster_centers_
                label_pred_zcommon = estimator.labels_


                # 计算不同视图样本的相似度，同簇的为正样本对，反之为负样本对
                # 1） 获得不同视图之间聚类中心的匹配关系矩阵 match_centriods

                # 其他视图的聚类中心均向 Z* 的聚类中心 靠齐, 得到不同视图向Z*对齐的中心分配结果
                h = torch.nn.functional.normalize(torch.from_numpy(centroids_views_zcommon), p=2, dim=1)
                # 每个superclass-K集合均考虑聚类中心之间的关系
                match_centriods = []
                match_centriods.append(list(range(curr_K[it])))# 第Z*视图的聚类中心作为标尺，其他进行靠近
                for v in range(view_num):
                    h_tmp = torch.nn.functional.normalize(torch.from_numpy(centroids_views[v]), p=2, dim=1)
                    Simialrity_centroids = h @ h_tmp.t() / config['training']['tau_cross']
                    # record the optimal pos_centriods,neg_centriods
                    row_ind, col_ind = linear_sum_assignment(Simialrity_centroids, maximize=True)  # 最大相似度
                    match_centriods.append(col_ind) # 得到不同视图向Z*对齐的中心分配结果

                # 2) 获得不同视图之间对应的下标。
                # 只计算 所有视图到公共视图之间的对比损失
                loss_cross_contrastive_value = 0
                h_latent_fusion_z_common = torch.nn.functional.normalize(latent_fusion_z_common, p=2, dim=1)

                for v in range(view_num):
                    h1 = torch.nn.functional.normalize(z[v], p=2, dim=1)
                    Simialrity_cross = h_latent_fusion_z_common @ h1.t() / config['training']['tau_cross']  # batch_size * batch_size (N * n_v)
                    # 同聚类簇的样本下标
                    label_view1_idx, label_view2_idx = [], []
                    for i in range(curr_K[it]):
                        label_view1_idx.append(np.where(label_pred_zcommon == match_centriods[0][i])[0])# 公共Z*的相同标签对应的样本下标
                        label_view2_idx.append(np.where(label_views[v][0] == match_centriods[v+1][i])[0])# 其他视图的相同标签对应的样本下标  # match_centriods包含了Z*的聚类中心，#label_views只有各视图
                    for i in range(Simialrity_cross.shape[0]):  # 当前视图样本的长度
                        # c = label_views[0][0][i]  # 当前样本类别为c,对应另一个视图的类别为clo_ind[i],下标为label_view2_idx[c]
                        c = label_pred_zcommon[i]
                        pos_cross_view = torch.sum(torch.exp(Simialrity_cross[i, label_view2_idx[c].tolist()]),
                                                   dim=0)
                        neg_cross_value = torch.sum(
                            torch.exp(Simialrity_cross[i, np.setdiff1d(np.array(range(Simialrity_cross.shape[1])),
                                                                       label_view2_idx[c])]), dim=0)
                        loss_cross_contrastive_value += - torch.log(pos_cross_view / neg_cross_value)
                structure_loss_cross_contrastive = loss_cross_contrastive_value / (nv * view_num)  # 20220612



                "4: cross-view guidance "
                '''稀疏化方式： 选择轮廓系数大于等于当前视图轮廓系数2倍的视图进行指导'''

                KL_divergence_loss = 0
                for v in range(view_num):
                    log_unreliable = F.log_softmax(commonZ_term_before[v], dim=1)  # for unreliable view
                    # reliable_view_idx = np.where(silhouette_score_avg >= silhouette_score_avg[v])
                    # reliable_view_idx = np.where(silhouette_score_avg >= silhouette_score_avg[v]*1.5)

                    a = sil_para * step_gamma ** epoch
                    a = 1 if a < 1 else a
                    reliable_view_idx = np.where(silhouette_score_avg >= silhouette_score_avg[v] * a)

                    reliable_view_idx_weight, reliable_view_idx_weight_tmp = [], []
                    for i in range(len(reliable_view_idx[0])):
                        reliable_view_idx_weight_tmp.append(self.sigmoid(silhouette_score_avg[reliable_view_idx[0][i]]))
                    # 归一化数组
                    reliable_view_idx_weight = self.normalize_rows(np.array([reliable_view_idx_weight_tmp]))
                    for v_t in range(len(reliable_view_idx[0])):
                        p_y = F.softmax(commonZ_term_before[reliable_view_idx[0][v_t]], dim=-1)  # # reliable view
                        KL_divergence_loss += reliable_view_idx_weight[0][v_t] * F.kl_div(log_unreliable, p_y,
                                                                                          reduction='sum')
                structure_loss_KL = structure_loss_KL + KL_divergence_loss


                loss = reconstruction_loss + loss_z_norm * config['training']['lambda1'][hyper_lambda1] + \
                       structure_loss_inner_contrastive * config['training']['lambda2'][hyper_lambda2] + \
                       structure_loss_cross_contrastive * config['training']['lambda3'][hyper_lambda3] + \
                       structure_loss_KL * config['training']['lambda4'][hyper_lambda4]


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_rec12 += recon1.item()+recon2.item()
                loss_z_norm1 += loss_z_norm.item() * config['training']['lambda1'][hyper_lambda1]
                loss_structure_inner_contrastive1 += structure_loss_inner_contrastive.item() * config['training']['lambda2'][hyper_lambda2] # 对比学习
                loss_structure_cross_contrastive1 += structure_loss_cross_contrastive.item() * config['training']['lambda3'][hyper_lambda3]  # 对比学习
                loss_structure_KL1 += structure_loss_KL * config['training']['lambda4'][hyper_lambda4] # KL divergence

            iter_current_epoch += 1
            '''# 每个epoch结束计算acc和nmi并记录'''
            '''用学得的autoencoder参数进行整体样本的评估，学得整体的子空间表示Z，然后用k-means得到Y_pre'''

            scores, latent_fusion, z, Y_list = self.evaluation_my_v3(config, logger, mask, train_view, Y_list, device,
                                           tmp_idx)

            acc_baseline, nmi_baseline, Precision_baseline, F_measure_baseline, recall_baseline, ARI_baseline, AMI_baseline \
                = scores['kmeans']['accuracy'], scores['kmeans']['NMI'], \
                  scores['kmeans']['precision'], scores['kmeans']['f_measure'], \
                  scores['kmeans']['recall'], scores['kmeans']['ARI'], scores['kmeans']['AMI']

            acc_show.extend([acc_baseline])
            nmi_show.extend([nmi_baseline])
            precision_show.extend([Precision_baseline])
            F_measure_show.extend([F_measure_baseline])
            recall_show.extend([recall_baseline])
            ARI_show.extend([ARI_baseline])
            AMI_show.extend([AMI_baseline])


        return scores['kmeans']['accuracy'], scores['kmeans']['NMI'], scores['kmeans']['precision'],\
               scores['kmeans']['f_measure'], scores['kmeans']['recall'], scores['kmeans']['ARI'], \
               scores['kmeans']['AMI'], acc_show, nmi_show, precision_show, F_measure_show, recall_show, ARI_show, AMI_show, d_A_distance_record



    def evaluation_my_v3(self, config, logger, mask, x_train, Y_list, device, tmp_idx):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            self.autoencoder3.eval(), self.autoencoder4.eval()
            self.autoencoder5.eval(), self.autoencoder6.eval()

            self.img2txt.eval(), self.txt2img.eval()
            view_num = len(x_train)
            train_view = x_train.copy()

            z = []
            z.append(self.autoencoder1.encoder(train_view[0]))
            z.append(self.autoencoder2.encoder(train_view[1]))
            z.append(self.autoencoder3.encoder(train_view[2]))
            z.append(self.autoencoder4.encoder(train_view[3]))
            z.append(self.autoencoder5.encoder(train_view[4]))
            z.append(self.autoencoder6.encoder(train_view[5]))

            latent_fusion = torch.cat([z[0], z[1], z[2], z[3], z[4], z[5], ], dim=0).cpu().numpy()  # 20220509

            scores = evaluation.clustering([latent_fusion], Y_list)
            logger.info("\033[2;29m" + 'view_concat ' + str(scores) + "\033[0m")

            self.autoencoder1.train(), self.autoencoder2.train()  # 观测样本
            self.autoencoder3.train(), self.autoencoder4.train()  # 观测样本
            self.autoencoder5.train(), self.autoencoder6.train()  # 观测样本

            self.img2txt.train(), self.txt2img.train()  # 缺失样本
        return scores, latent_fusion, z, Y_list



