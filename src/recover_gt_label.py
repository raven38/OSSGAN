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

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config_path', type=str, default='./src/configs/CIFAR10/ContraGAN.json')
    parser.add_argument('src', type=str)
    parser.add_argument('dst', type=str)
    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_configs = json.load(f)
    else:
        raise NotImplementedError

    src, dst = args.src, args.dst

    config = model_configs['data_processing']

    assert Path(src).exists(), 'src does not exists'
    assert Path(dst).exists(), 'dst does not exists'

    assert os.path.splitext(src)[-1] == '.hdf5', 'only accept hdf5 file'
    assert os.path.splitext(dst)[-1] == '.hdf5', 'only accept hdf5 file'

    with h5.File(src, 'r') as f:
        imgs = f['imgs'][...]
        labels = f['labels'][...]

        assert len(imgs.shape) == 4
        assert len(imgs) == len(labels)

    with h5.File(dst, 'a') as f:
        dst_imgs = f['imgs'][...]
        dst_labels = f['labels'][...]
        assert np.array_equal(dst_labels[dst_labels != -1], labels[dst_labels != -1])
        assert np.array_equal(f['imgs'][...], imgs)

        labels2 = f.get('labels2')
        if labels2:
            assert np.array_equal(labels, labels2[...])
        else:
            f.create_dataset('labels2', data=labels, dtype='int64', maxshape=labels.shape,
                             chunks=(config['chunk_size'],), compression=config['compression'])
