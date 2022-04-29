# OSSGAN: Open-Set Semi-Supervised Image Generation
### [CVPR 2022] Official pytorch implementation

### Prepare envrioment


To run the code, you need pytorch and some additional packages.
```
conda env create env.xml
conda activate torch
```




# Quick Start

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPU ``0``
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -e -c CONFIG_PATH
```

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPUs ``(0, 1, 2, 3)`` and ``DataParallel``
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -c CONFIG_PATH
```

Try ``python3 src/main.py`` to see available options.


# ImageNet

### Prepare dataset
Manual download of the ImageNet dataset (for evaluation and training). 
Please follow the instructions https://www.tensorflow.org/datasets/catalog/imagenet2012

Put the training and validation set of the ImageNet dataset on `./code/data/ILSVRC2012/{train|valid}`.

```
python3 src/main.py -t -e -l -s -iv -sync_bn -stat_otf -mpc --eval_type valid -c src/configs/ILSVRC2012/BigGAN256.json
```


Make dataset

```
python3 src/make_osssimagenet.py --src data/ILSVRC2012 --dst data/OSSSILSVRC2012_50_020_010 --subset_class 50 --ratio 0.2 --usage 0.1
```

### Training

```
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -e -l -sync_bn -stat_otf --eval_type valid -c CONFIG_PATH
```

### Testing

```
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -e -l -sync_bn -stat_otf --eval_type valid -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER
```

# Tiny ImageNet

### prepare dataset
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip 
unzip tiny-imagenet-200.zip
```

Put the training and validation set to `code/data/TINY_ILSVRC2012/{train|valid}`

```
python3 src/main.py -t -e -l -s -iv -sync_bn -stat_otf -mpc --eval_type valid -c src/configs/TINY_ILSVRC2012/BigGAN-Mod.json
```

Make datasets

```
python3 src/make_semi_supervised_dataset.py --src data/TINY_ILSVRC2012 --dst data/OSSSTINY_ILSVRC2012_50_010 --subset_class 50 --ratio 0.1 
```


Run dataset

```

python3 src/main.py -t -e -l -sync_bn -stat_otf --eval_type valid -c CONFIG_PATH
```


## License
This repo is built on top of [StudioGAN][https://github.com/POSTECH-CVLab/PyTorch-StudioGAN]. 
However, portions of the library are avaiiable under distinct license terms: StyleGAN2, StyleGAN2-ADA, and StyleGAN3 are licensed under [NVIDIA source code license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE-NVIDIA), Synchronized batch normalization is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/sync_batchnorm/LICENSE), HDF5 generator is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/hdf5.py), differentiable SimCLR-style augmentations is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/simclr_aug.py), and clean-FID is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/resize.py).

## Bibtex
```

```
