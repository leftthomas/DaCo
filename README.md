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
from [MEGA](https://mega.nz/folder/kx53iYoL#u_Zc6ogPokaTRVM6qYn3ZA). The data should be rearranged, please refer the
paper to acquire the details of `train/val` split. The data directory structure is shown as follows:

 ```
├──tokyo
   ├── original (orignal images)
       ├── domain_a (day images)
           ├── train
               ├── day_00001.png
               └── ...
           ├── val
               ├── day_00301.png
               └── ...
       ├── domain_b (night images)
           ├── train
               ├── night_00001.png
               └── ...
           ├── val
               ├── night_00301.png
               └── ...
   ├── generated (generated images)
       same structure as original
       ...
├──cityscapes
   same structure as tokyo
   ...
├──alderley
   same structure as tokyo 
   ... 
```

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

The models are trained on one NVIDIA GTX TITAN (12G) GPU. `Adam` is used to optimize the model, `lr` is `1e-3`
and `weight decay` is `1e-6`. `batch size` is `16` for `daco`, `32` for `simclr` and `moco`, `64` for `npid`.
`momentum` is `0.999` for `moco`, other hyper-parameters are the default values.

### Tokyo 24/7
<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">Day --&gt; Night</th>
    <th colspan="4">Night --&gt; Day</th>
    <th colspan="4">Day &lt;--&gt; Night</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td>R@1</td>
    <td>R@2</td>
    <td>R@4</td>
    <td>R@8</td>
    <td>R@1</td>
    <td>R@2</td>
    <td>R@4</td>
    <td>R@8</td>
    <td>R@1</td>
    <td>R@2</td>
    <td>R@4</td>
    <td>R@8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>NPID</td>
    <td>8.00</td>
    <td>9.33</td>
    <td>10.67</td>
    <td>14.67</td>
    <td>8.00</td>
    <td>9.33</td>
    <td>10.67</td>
    <td>12.00</td>
    <td>3.33</td>
    <td>6.00</td>
    <td>6.67</td>
    <td>8.00</td>
    <td>abcd</td>
  </tr>
  <tr>
    <td>MoCo</td>
    <td>5.33</td>
    <td>5.33</td>
    <td>8.00</td>
    <td>17.33</td>
    <td>6.67</td>
    <td>8.00</td>
    <td>12.00</td>
    <td>21.33</td>
    <td>0.00</td>
    <td>0.00</td>
    <td>0.00</td>
    <td>0.67</td>
    <td>efgh</td>
  </tr>
  <tr>
    <td>SimCLR</td>
    <td>29.33</td>
    <td>33.33</td>
    <td>45.33</td>
    <td>58.67</td>
    <td>32.00</td>
    <td>40.00</td>
    <td>46.67</td>
    <td>57.33</td>
    <td>6.00</td>
    <td>10.00</td>
    <td>14.00</td>
    <td>20.00</td>
    <td>hhhh</td>
  </tr>
  <tr>
    <td>DaCo</td>
    <td>69.33</td>
    <td>73.33</td>
    <td>81.33</td>
    <td>88.00</td>
    <td>65.33</td>
    <td>80.00</td>
    <td>85.33</td>
    <td>90.67</td>
    <td>52.00</td>
    <td>60.67</td>
    <td>73.33</td>
    <td>81.33</td>
    <td>rrrr</td>
  </tr>
</tbody>
</table>