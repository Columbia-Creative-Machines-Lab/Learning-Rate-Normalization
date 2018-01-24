# Clonning the densenet Repo
git clone https://github.com/gpleiss/efficient_densenet_pytorch
cd efficient_densenet_pytorch
rm demo.py
wget https://raw.githubusercontent.com/Columbia-Creative-Machines-Lab/Learning-Rate-Normalization/master/sgd_lr_norm.py -O sgd_lr_norm.py
wget https://raw.githubusercontent.com/Columbia-Creative-Machines-Lab/Learning-Rate-Normalization/master/densenet_demo.py -O demo.py
pip3 install fire

# Baselines
# SGD
sed -i '/optimizer = optim./c\    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)' demo.py
echo "SGD_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data cifar10 --save SGD_CIFAR | tee sgd_cifar.txt
echo "SGD_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data svhn --save SGD_SVHN | tee sgd_svhn.txt

# SGD+Momentum 
sed -i '/optimizer = optim./c\    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)' demo.py
echo "SGD_momentum_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data ciraf10 --save SGD_momentum_CIFAR | tee sgd_momentum_cifar.txt
echo "SGD_momentum_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data svhn --save SGD_momentum_SVHN | tee sgd_momentum_svhn.txt

# Adam
sed -i '/optimizer = optim./c\    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)' demo.py
echo "Adam_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data cifar10--save Adam_CIFAR | tee adam_cifar.txt
echo "Adam_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data svhn --save Adam_SVHN | tee adam_svhn.txt

# RMSProp
sed -i '/optimizer = optim./c\    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)' demo.py
echo "RMSPROP_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data cifar10--save RMSProp_CIFAR | tee rmsprop_cifar.txt
echo "RMSPROP_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data svhn --save RMSProp_SVHN | tee rmsprop_svhn.txt

# Adadelta
sed -i '/optimizer = optim./c\    optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=wd)' demo.py
echo "Adadelta_CIFAR"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data cifar10--save Adadelta_CIFAR | tee adadelta_cifar.txt
echo "Adadelta_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data svhn --save Adadelta_SVHN | tee adadelta_svhn.txt

# SGD_LR_Norm
echo "SGD_lr_norm_momentum_CIFAR"
sed -i '/optimizer = optim./c\    optimizer = optim.SGD_lr_norm(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)' demo.py
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data cifar10--save SGD_lr_norm_momentum_CIFAR | tee sgd_lr_norm_cifar.txt
echo "SGD_lr_norm_momentum_SVHN"
CUDA_VISIBLE_DEVICES=0 python3 demo.py --efficient True --data svhn --save SGD_lr_norm_momentum_SVHN | tee sgd_lr_norm_svhn.txt

echo "Densenet experiments finished"
