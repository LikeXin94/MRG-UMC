Title: Multi-level Reliable Guidance for Unpaired Multi-view Clustering

To illustrate the implementation, the code takes the digit dataset with six-view as an example. The main file of the code is train_MRG_UMC_6views.py.

1. Requirements:
pytorch==1.2.0
numpy>=1.19.1
scikit-learn>=0.23.2
munkres>=1.1.4

2. Configuration: The hyper-parameters are defined in configure.py.

3. Datasets: The Digit[1], Caltech101-20[2], Scene-15[2], Flower17[3] and Reuters[4] datasets are list as follows:

[1] http://archive.ics.uci.edu/dataset/72/multiple+features

[2] https://github.com/XLearning-SCU/2021-CVPR-Completer/tree/main/data

[3] http://www.robots.ox.ac.uk/vgg/data/flowers/17/index.html

[4] http://archive.ics.uci.edu/ml/datasets/Reuters+RCV1+RCV2+Multilingual%2C+Multiview+Text+Categorization+Test+collection
