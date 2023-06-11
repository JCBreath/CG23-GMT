# GMT: A deep learning approach to generalized multivariate translation for scientific data analysis and visualization

PyTorch implementation of the paper _GMT: A deep learning approach to generalized multivariate translation for scientific data analysis and visualization_

## Prerequisite

* Linux
* Python 3.7+
* CUDA 11.3+
* PyTorch 1.11.0+
* NumPy
* RAM 32GB+
* VRAM 16GB+

## Data Format
The volume contains little-endian floats in column-major order (z-axis, y-axis, x-axis).

## Training
Pretrain GMT for 500 epochs  
```
python3 pretrain.py --data_path /your/data/path --model_path /your/model/path --max_epoch 500 --dataset dataset_name
```

Fine-tune GMT for 4000 epochs  
```
python3 train.py --data_path /your/data/path --model_path /your/model/path --max_epoch 4000 --dataset dataset_name
```

Inference a trained GMT model (translating 0th variable to 1st variable)
```
python3 inference.py --data_path /your/data/path --model_path /your/model/path --epoch 4000 --dataset dataset_name --source 0 --target 1
```

## Citation
```
@article{Yao-GMT-CG23,
title = {GMT: A deep learning approach to generalized multivariate translation for scientific data analysis and visualization},
journal = {Computers & Graphics},
volume = {112},
pages = {92-104},
year = {2023},
author = {Siyuan Yao and Jun Han and Chaoli Wang}
}
```
