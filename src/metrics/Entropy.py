# src/metrics/Accuracy.py


import numpy as np
import math
from scipy import linalg
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise
from utils.predict import pred_cls_out

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def calculate_entropy(dataloader, generator, discriminator, D_loss, num_evaluate, truncated_factor, prior, latent_op,
                      latent_op_step, latent_op_alpha, latent_op_beta, device, cr, logger):
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size
    disable_tqdm = device != 0

    if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel):
        z_dim = generator.module.z_dim
        num_classes = generator.module.num_classes
        conditional_strategy = discriminator.module.conditional_strategy
    else:
        z_dim = generator.z_dim
        num_classes = generator.num_classes
        conditional_strategy = discriminator.conditional_strategy

    total_batch = num_evaluate//batch_size

    if device == 0: logger.info("Calculate Entropy....")

    confid = []
    confid_label = []
    confid_label2 = []
    for batch_id in tqdm(range(total_batch), disable=disable_tqdm):
        try:
            real_images, real_labels, *other = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_images, real_labels, *other = next(data_iter)
        real_images, real_labels = real_images.to(device), real_labels.to(device)

        if len(other) == 0:
            return 0, 0, 0
        labels2 = other[0]
        with torch.no_grad():
            try:
                cls_out_real = pred_cls_out(discriminator, real_images, real_labels, conditional_strategy).float()
            except NotImplementedError:
                return 0, 0, 0 

            prob = F.softmax(cls_out_real, dim=1)
            entropy = (-1 * (prob * torch.log(prob)).sum(1)).detach().cpu().numpy() / np.log(num_classes)
            real_labels = real_labels.detach().cpu().numpy()
            labels2 = labels2.detach().cpu().numpy()

        confid.append(entropy)
        confid_label.append(real_labels)
        confid_label2.append(labels2)

    confid = np.concatenate(confid, axis=0)
    confid_label = np.concatenate(confid_label, axis=0)
    confid_label2 = np.concatenate(confid_label2, axis=0)

    # open_entropy = confid[confid_label == -1].mean()
    # close_entropy = confid[confid_label != -1].mean()
    labeled_entropy = confid[confid_label != -1].mean()
    unlabeled_open_entropy = confid[(confid_label == -1) & (confid_label2 < num_classes)].mean()
    unlabeled_close_entropy = confid[(confid_label == -1) & (confid_label2 >= num_classes)].mean()

    return labeled_entropy, unlabeled_close_entropy, unlabeled_open_entropy
