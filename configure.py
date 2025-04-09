#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'xlk'
#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'xlk'
def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=1,#0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=200,
                lr=1e-4,  # [1.0e-4],
                # super-class 最优参数
                lambda1=[1e-4],  # Z-norm; Orthogonal constrain
                lambda2=[1e-4],  # inner-view multi-level clustering
                lambda3=[1e-4],  # synthesized-view alignment
                lambda4=[1e4],  # cross-view guidance
                tau_inter=1e-1,  # the temperature parameter of inner-view multi-level clustering
                tau_cross=1e-1,  # the temperature parameter of synthesized-view alignment

                # 其余参数（lambda5-8）不用
                lambda5=[0],
                lambda6=0,
                lambda7=[0],
                lambda8=[0],
            ),
        )


    elif data_name in ['digit_2view']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[76, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,
                epoch=200,
                lr=1e-4,
                alpha=[0.9],
                # super-class 最优参数
                lambda1=[1e-4],  # Z-norm; Orthogonal constrain
                lambda2=[1e-4],  # inner-view multi-level clustering
                lambda3=[1e-4],  # synthesized-view alignment
                lambda4=[1e4],  # cross-view guidance
                tau_inter=1e-1,  # the temperature parameter of inner-view multi-level clustering
                tau_cross=1e-1,  # the temperature parameter of synthesized-view alignment

                # 其余参数（lambda5-8）不用
                lambda5=[0],
                lambda6=0,
                lambda7=[0],
                lambda8=[0],
            ),
        )



    elif data_name in ['Caltech101-20-6view']:
        """The default configs."""  # #[48,40,254,1984,512] [2358]
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
                arch6=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[48, 1024, 1024, 1024, 128],
                arch2=[40, 1024, 1024, 1024, 128],
                arch3=[254, 1024, 1024, 1024, 128],
                arch4=[1984, 1024, 1024, 1024, 128],
                arch5=[512, 1024, 1024, 1024, 128],
                arch6=[928, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                activations6='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,  # 0.5,
                seed=8,
                batch_size=256,
                epoch=200,
                lr=1e-4,
                alpha=[0.001],
                # super-class 最优参数
                lambda1=[1e-2],  # Z-norm; Orthogonal constrain
                lambda2=[1e-3],  # inner-view multi-level clustering
                lambda3=[1e-3],  # synthesized-view alignment
                lambda4=[1e4],  # cross-view guidance
                tau_inter=1e-1,  # the temperature parameter of inner-view multi-level clustering
                tau_cross=1e-1,  # the temperature parameter of synthesized-view alignment

                # 其余参数（lambda5-8）不用
                lambda5=[0],
                lambda6=0,
                lambda7=[0],
                lambda8=[0],

            ),
        )



    elif data_name in ['digit_6view']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
                arch6=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[76, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                arch3=[64, 1024, 1024, 1024, 128],
                arch4=[240, 1024, 1024, 1024, 128],
                arch5=[47, 1024, 1024, 1024, 128],
                arch6=[6, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                activations6='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,
                epoch=200,
                lr=1e-4,
                alpha=[0.001],
                # super-class 最优参数
                lambda1=[1e-4],  # Z-norm; Orthogonal constrain
                lambda2=[1e-4],  # inner-view multi-level clustering
                lambda3=[1e-4],  # synthesized-view alignment
                lambda4=[1e4],  # cross-view guidance
                tau_inter=1e-1,  # the temperature parameter of inner-view multi-level clustering
                tau_cross=1e-1,  # the temperature parameter of synthesized-view alignment

                # 其余参数（lambda5-8）不用
                lambda5=[0],
                lambda6=0,
                lambda7=[0],
                lambda8=[0],

            ),
        )



    else:
        raise Exception('Undefined data_name')
