"""
MIT License

Copyright (c) 2021 Kai Katsumata
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import h5py as h5
import numpy as np
import PIL
import json
from argparse import ArgumentParser
from functools import partial
import shutil
from glob import glob
from pathlib import Path

# from utils.make_hdf5 import make_hdf5, make_subset_hdf5


def osss_subset(labels, ratio=0.2, subset_class=10):
    N = len(labels)
    num_classes = len(np.unique(labels))
    assert subset_class < num_classes

    class_mask = np.where(subset_class <= labels, 1, 0)
    unlabeled_mask = np.random.choice([True, False], size=N, p=[1 - ratio, ratio])
    unlabeled_mask = unlabeled_mask | class_mask
    labels = labels * (1 - unlabeled_mask) + (-1) * unlabeled_mask
    return labels


def ss_subset(labels, ratio=0.2):
    N = len(labels)
    unlabeled_mask = np.random.choice([True, False], size=N, p=[1 - ratio, ratio])
    labels = labels * (1 - unlabeled_mask) + (-1) * unlabeled_mask
    return labels


def class_filtered_dataset(imgs, labels, subset_classes=[0]):
    mask = np.isin(labels,  subset_classes)
    return imgs[mask], labels[mask]


def make_semi_supervised_dataset(hdf5_path, config, sub_f):
    with h5.File(hdf5_path, 'a') as f:
        labels = f['labels'][...]
        ss_labels = sub_f(labels)
        assert labels.shape == ss_labels.shape
        olabels_dset = f.create_dataset('original_labels', labels.shape, dtype='int64', maxshape=labels.shape,
                                        chunks=(config['chunk_size'],), compression=config['compression'])
        olabels_dset[...] = labels
        data = f['labels']
        data[...] = ss_labels    


def make_filtered_dataset(hdf5_path, config, filter_f):
    with h5.File(hdf5_path, 'a') as f:
        labels = f['labels'][...]
        imgs = f['imgs'][...]
        ss_imgs, ss_labels = filter_f(imgs, labels)
        f.create_dataset('original_labels', data=labels, dtype='int64', maxshape=labels.shape,
                         chunks=(config['chunk_size'],), compression=config['compression'])
        del f['imgs']
        del f['labels']
        f.create_dataset('imgs', data=ss_imgs, dtype='uint8',
                         chunks=(config['chunk_size'], 3, config['img_size'], config['img_size']), compression=config['compression'])
        f.create_dataset('labels', data=ss_labels, dtype='int64', maxshape=labels.shape,
                         chunks=(config['chunk_size'],), compression=config['compression'])
        # data = f['labels']
        # data[...] = ss_labels
        # data = f['imgs']
        # data[...] = ss_imgs


def report_stats(hdf5_path):
    with h5.File(hdf5_path, 'r') as f:
        labels = f['labels'][...]
        imgs = f['imgs'][...]
        print(f'label shape: {labels.shape}')
        print(f'image shape: {imgs.shape}')
        print(f'The number of classes: {len(np.unique(labels))}, max: {np.max(labels)}, min: {np.min(labels)}')


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config_path', type=str, default='./src/configs/CIFAR10/ContraGAN.json')
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--subset_class', type=int, default=-1)
    parser.add_argument('--ratio', type=float, default=0.2, help='The ratio of remaining labels')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    args = parser.parse_args()

    assert args.src and args.dst, 'argument src and dst is needed'

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_configs = json.load(f)
        train_configs = vars(args)
    else:
        raise NotImplementedError

    assert not Path(args.dst).exists(), 'args.dst is existing path'
    shutil.copytree(args.src, args.dst)
    src_path_list = list(glob(f'{args.dst}/*train*.hdf5'))
    assert len(src_path_list) == 1, 'There are some datasets for train'
    hdf5_path_train = src_path_list[0]
    # hdf5_path_train = make_hdf5(model_configs['data_processing'], train_configs, mode="train")

    if args.subset_class != -1:
        sub_f = partial(osss_subset, ratio=args.ratio, subset_class=args.subset_class)
    else:
        sub_f = partial(ss_subset, ratio=args.ratio)

    make_semi_supervised_dataset(hdf5_path_train, model_configs['data_processing'], sub_f)
    report_stats(hdf5_path_train)
    src_path_list = list(glob(f'{args.dst}/*eval*.hdf5'))
    assert len(src_path_list) == 1, 'There are some datasets for eval'
    hdf5_path_eval = src_path_list[0]
    if args.subset_class != -1:
        make_filtered_dataset(hdf5_path_eval, model_configs['data_processing'],
                              partial(class_filtered_dataset, subset_classes=list(range(args.subset_class))))
    report_stats(hdf5_path_eval)
    # if args.subset_class != -1:
    #     _ = make_subset_hdf5(model_configs['data_processing'], train_configs, mode="eval", subset_class=args.subset_class)
    # else:
    #     _ = make_hdf5(model_configs['data_processing'], train_configs, mode="eval")
