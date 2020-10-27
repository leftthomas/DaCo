# DaCo
A PyTorch implementation of DaCo based on CVPR 2021 paper [DaCo: Domain-agnostic Contrastive Learning for Visual Localization]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch
```

## Dataset
`Alderley` and `CMU Season` datasets are used in this repo.

## Usage
```
python train.py --epochs 50 --feature_dim 256
optional arguments:
--feature_dim                 Feature dim for each image [default value is 128]
--m                           Negative sample number [default value is 4096]
--temperature                 Temperature used in softmax [default value is 0.1]
--momentum                    Momentum used for the update of memory bank [default value is 0.5]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 128]
--epochs                      Number of sweeps over the dataset to train [default value is 200]
```
