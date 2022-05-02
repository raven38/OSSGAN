# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/model_ops.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn import init


def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.ConvTranspose2d)
                or isinstance(module, nn.Linear)):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                print('Init style not recognized...')
        elif isinstance(module, nn.Embedding):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        else:
            pass


def get_activation_fn(activation_fn):
    act_fn = {
        "ReLU": nn.ReLU(inplace=True),
        "Leaky_ReLU": nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "ELU": nn.ELU(alpha=1.0, inplace=True),
        "GELU": nn.GELU()
    }
    try:
        return act_fn[activation_fn]
    except KeyError:
        raise NotImplementedError


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

def embedding(num_embeddings, embedding_dim):
    return nn.Linear(in_features=num_embeddings, out_features=embedding_dim, bias=False)

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), eps=1e-6)

def sndeconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), eps=1e-6)

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Linear(in_features=num_embeddings, out_features=embedding_dim, bias=False), eps=1e-6)

def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)


def snembeddingent(num_embeddings, embedding_dim):
    return spectral_norm(LinearEnt(in_features=num_embeddings+1, out_features=embedding_dim, bias=False), eps=1e-6)


def embeddingent(num_embeddings, embedding_dim):
    return LinearEnt(in_features=num_embeddings+1, out_features=embedding_dim, bias=False)


class ConditionalHead(nn.Module):
    def __init__(self, in_dim, d_spectral_norm, conditional_strategy, hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, softmax_threshold):
        super(ConditionalHead, self).__init__()
        self.conditional_strategy = conditional_strategy

        which_linear = snlinear if d_spectral_norm else linear
        which_embedding = sn_embedding if d_spectral_norm else embedding

        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.conditional_strategy = conditional_strategy
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.threshold = softmax_threshold

        self.linear1 = which_linear(in_features=self.in_dim, out_features=1)
        if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
            self.linear2 = which_linear(in_features=self.in_dim, out_features=hypersphere_dim)
            if self.nonlinear_embed:
                self.linear3 = which_linear(in_features=hypersphere_dim, out_features=hypersphere_dim)
            self.embedding = which_embedding(num_classes, hypersphere_dim)
        elif self.conditional_strategy in ['ProjGAN', 'Random']:
            self.embedding = which_embedding(num_classes, self.in_dim)
        elif self.conditional_strategy in ['Single']:
            self.embedding = nn.Sequential(which_linear(in_features=num_classes+1, out_features=self.in_dim*2),
                                           which_linear(in_features=self.in_dim*2, out_features=self.in_dim))
        elif self.conditional_strategy == 'ACGAN':
            self.linear4 = which_linear(in_features=self.in_dim, out_features=num_classes)
        elif self.conditional_strategy in ['SSGAN', 'OSSSGAN', 'Reject']:
            self.embedding = which_embedding(num_classes, self.in_dim)
            self.linear4 = which_linear(in_features=self.in_dim, out_features=num_classes)
        elif self.conditional_strategy in ['Open']:
            self.embedding = which_embedding(num_classes+1, self.in_dim)
            self.linear4 = which_linear(in_features=self.in_dim, out_features=num_classes)
        else:
            pass

    def forward(self, h, label_):
        label = torch.eye(self.num_classes, device=label_.device)[label_]

        if self.conditional_strategy == 'no':
            authen_output = torch.squeeze(self.linear1(h))
            return authen_output

        elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
            authen_output = torch.squeeze(self.linear1(h))
            cls_proxy = self.embedding(label)
            cls_embed = self.linear2(h)
            if self.nonlinear_embed:
                cls_embed = self.linear3(self.activation(cls_embed))
            if self.normalize_embed:
                cls_proxy = F.normalize(cls_proxy, dim=1)
                cls_embed = F.normalize(cls_embed, dim=1)
            return cls_proxy, cls_embed, authen_output

        elif self.conditional_strategy == 'ProjGAN':
            index = torch.nonzero(label_ != -1).squeeze()
            authen_output = torch.squeeze(self.linear1(h.index_select(0, index)))
            proj = torch.sum(torch.mul(self.embedding(label.index_select(0, index)), h.index_select(0, index)), 1)
            return proj + authen_output

        elif self.conditional_strategy == 'Random':
            authen_output = torch.squeeze(self.linear1(h))
            mask = (label_ == -1).int()
            random_label = torch.eye(self.num_classes, device=label_.device)[torch.randint(0, self.num_classes, label_.shape, device=label_.device)]
            label2 = label * (1 - mask).view(-1, 1) + random_label * mask.view(-1, 1)
            proj = torch.sum(torch.mul(self.embedding(label2), h), 1)
            return proj + authen_output

        elif self.conditional_strategy == 'Single':
            authen_output = torch.squeeze(self.linear1(h))
            label = torch.eye(self.num_classes + 1, device=label_.device)[label_]
            proj = torch.sum(torch.mul(self.embedding(label), h), 1)
            return proj + authen_output

        elif self.conditional_strategy == 'ACGAN':
            authen_output = torch.squeeze(self.linear1(h))
            cls_output = self.linear4(h)
            return cls_output, authen_output

        elif self.conditional_strategy == 'SSGAN':
            authen_output = torch.squeeze(self.linear1(h))
            cls_output = self.linear4(h)
            mask = (label_ == -1).int()
            label2 = label * (1 - mask).view(-1, 1) + F.softmax(cls_output, dim=1) * mask.view(-1, 1)
            proj = torch.sum(torch.mul(self.embedding(label2.detach()), h), 1)
            return cls_output, proj + authen_output

        elif self.conditional_strategy == 'Open':
            label = torch.eye(self.num_classes + 1, device=label_.device)[label_]
            authen_output = torch.squeeze(self.linear1(h))
            cls_output = self.linear4(h)
            open_mask = (label_ == -1) & (torch.max(F.softmax(cls_output.detach()), dim=1)[0] < self.threshold)
            pred = torch.argmax(cls_output, dim=1)
            assert open_mask.shape == pred.shape
            pred[open_mask] = self.num_classes
            pred_label = torch.eye(self.num_classes + 1, device=label_.device)[pred]
            assert pred_label.shape == (len(label), self.num_classes+1)
            mask = (label_ == -1).int()
            label2 = label * (1 - mask).view(-1, 1) + pred_label * mask.view(-1, 1)
            proj = torch.sum(torch.mul(self.embedding(label2.detach()), h), 1)
            return cls_output, proj + authen_output

        elif self.conditional_strategy == 'Reject':
            cls_output = self.linear4(h)
            index = ((label_ != -1) | (torch.max(F.softmax(cls_output.detach(), dim=1), dim=1)[0] >= self.threshold)).nonzero().squeeze()

            authen_output = torch.squeeze(self.linear1(h.index_select(0, index)))
            mask = (label_ == -1).int()
            label2 = label * (1 - mask).view(-1, 1) + torch.eye(self.num_classes, device=label_.device)[torch.argmax(cls_output, dim=1)] * mask.view(-1, 1)
            proj = torch.sum(torch.mul(self.embedding(label2.index_select(0, index).detach()), h.index_select(0, index)), 1)
            return cls_output, proj + authen_output

        elif self.conditional_strategy == 'OSSSGAN':
            authen_output = torch.squeeze(self.linear1(h))
            cls_output = self.linear4(h)
            mask = (label_ == -1).int()
            label2 = label * (1 - mask).view(-1, 1) + F.softmax(cls_output, dim=1) * mask.view(-1, 1)
            proj = torch.sum(torch.mul(self.embedding(label2.detach()), h), 1)
            return cls_output, proj + authen_output

        else:
            raise NotImplementedError


class LinearEnt(nn.Linear):
    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1, keepdim=True)
        h = torch.cat((x, b), dim=1)
        return super().forward(h)


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, num_features, num_classes, spectral_norm):
        super().__init__()
        self.num_features = num_features
        self.bn = batchnorm_2d(num_features, eps=1e-4, momentum=0.1, affine=False)

        if spectral_norm:
            self.embed0 = sn_embedding(num_classes, num_features)
            self.embed1 = sn_embedding(num_classes, num_features)
        else:
            self.embed0 = embedding(num_classes, num_features)
            self.embed1 = embedding(num_classes, num_features)

    def forward(self, x, y):
        gain = (1 + self.embed0(y)).view(-1, self.num_features, 1, 1)
        bias = self.embed1(y).view(-1, self.num_features, 1, 1)
        out = self.bn(x)
        return out * gain + bias


class ConditionalBatchNorm2d_for_skip_and_shared(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, num_features, z_dims_after_concat, spectral_norm):
        super().__init__()
        self.num_features = num_features
        self.bn = batchnorm_2d(num_features, eps=1e-4, momentum=0.1, affine=False)

        if spectral_norm:
            self.gain = snlinear(z_dims_after_concat, num_features, bias=False)
            self.bias = snlinear(z_dims_after_concat, num_features, bias=False)
        else:
            self.gain = linear(z_dims_after_concat, num_features, bias=False)
            self.bias = linear(z_dims_after_concat, num_features, bias=False)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias


class Self_Attn(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, in_channels, spectral_norm):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels

        if spectral_norm:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma*attn_g
