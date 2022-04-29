def pred_dis_out(net, x, label_, conditional_strategy, evaluation=False):
    outputs = net(x, label_, evaluation=evaluation)
    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, dis_out = outputs
    elif conditional_strategy in ['ProjGAN', 'no', 'Random', 'Single']:
        dis_out = outputs
    elif conditional_strategy in ['ACGAN', 'SSGAN', 'OSSSGAN', 'Open', 'Reject']:
        _, dis_out = outputs
    else:
        raise NotImplementedError

    return dis_out


def pred_cls_out(net, x, label_, conditional_strategy, evaluation=False):
    if conditional_strategy in ['ACGAN', 'SSGAN', 'OSSSGAN', 'Open', 'Reject']:
        cls_out, _ = net(x, label_, evaluation=evaluation)
    else:
        raise NotImplementedError

    return cls_out
