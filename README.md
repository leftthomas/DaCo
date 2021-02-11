# DaCo
A PyTorch implementation of DaCo based on ICMR 2021
paper [DaCo: Domain-Agnostic Contrastive Learning for Visual Localization]().

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

## Dataset

[Tokyo 24/7](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/)
, [Cityscapes FoggyDBF](https://www.cityscapes-dataset.com)
and [Alderley](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395) datasets are used in this repo, you could
download these datasets from official websites, or download them
from [MEGA](https://mega.nz/folder/kx53iYoL#u_Zc6ogPokaTRVM6qYn3ZA). The data should be arranged like this, please refer
the paper to acquire the details of `train/val` split.

## Usage

```
python main.py --data_name alderley --method_name simclr --gpu_ids 0 1
optional arguments:
--data_root                   Datasets root path [default value is 'data']
--data_name                   Dataset name [default value is 'tokyo'](choices=['tokyo', 'cityscapes', 'alderley'])
--method_name                 Method name [default value is 'daco'](choices=['daco', 'simclr', 'moco', 'npid'])
--proj_dim                    Projected feature dim for computing loss [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.1]
--batch_size                  Number of images in each mini-batch [default value is 16]
--iters                       Number of bp over the model to train [default value is 10000]
--gpu_ids                     Selected gpus to train [required]  
--ranks                       Selected recall [default value is '1,2,4,8']
--save_root                   Result saved root path [default value is 'result']
--lamda                       Lambda used for the weight of soft constrain [default value is 0.8]
--negs                        Negative sample number [default value is 4096]
--momentum                    Momentum used for the update of memory bank or shadow model [default value is 0.5]
```

For example, to train `moco` on `cityscapes`:

```
python main.py --data_name cityscapes --method_name moco --batch_size 32 --gpu_ids 1 --momentum 0.999
```

to train `npid` on `tokyo`:

```
python main.py --data_name tokyo --method_name npid --batch_size 64 --gpu_ids 2 --momentum 0.5
```

## Results
