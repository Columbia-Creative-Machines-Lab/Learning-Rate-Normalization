# Clonning the densenet Repo
git clone https://github.com/bamos/densenet.pytorch
pip3 install setproctitle, graphviz
cd densenet.pytorch
wget https://raw.githubusercontent.com/Columbia-Creative-Machines-Lab/Learning-Rate-Normalization/master/sgd_lr_norm.py -O sgd_lr_norm.py
wget https://raw.githubusercontent.com/Columbia-Creative-Machines-Lab/Learning-Rate-Normalization/master/adam_lr_norm.py -O adam_lr_norm.py
rm train.py
wget https://raw.githubusercontent.com/Columbia-Creative-Machines-Lab/Learning-Rate-Normalization/master/densenet_train.py -O train.py

# Baselines
# SGD+Momentum
echo "SGD_momentum_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt sgd --data cifar | tee sgd_momentum_cifar.txt
echo "SGD_momentum_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt sgd --data svhn | tee sgd_momentum_svhn.txt

# Adam
echo "Adam_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt adam --data cifar | tee adam_cifar.txt
echo "Adam_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt adam --data svhn | tee adam_svhn.txt

# RMSProp
echo "RMSPROP_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt rmsprop --data cifar | tee rmsprop_cifar.txt
echo "RMSPROP_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt rmsprop --data svhn | tee rmsprop_svhn.txt

# Adadelta
echo "Adadelta_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt adadelta --data cifar | tee adadelta_cifar.txt
echo "Adadelta_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt adadelta --data svhn | tee adadelta_svhn.txt

# SGD_LR_Norm
echo "SGD_lr_norm_momentum_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt sgd_lr_norm --data cifar | tee sgd_lr_norm_cifar.txt
echo "SGD_lr_norm_momentum_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt sgd_lr_norm --data svhn | tee sgd_lr_norm_svhn.txt

# Adam_LR_Norm
echo "Adam_lr_norm_momentum_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt adam_lr_norm --data cifar | tee adam_lr_norm_cifar.txt
echo "Adam_lr_norm_momentum_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 train.py --opt adam_lr_norm --data svhn | tee adam_lr_norm_svhn.txt

echo "Densenet experiments finished"
