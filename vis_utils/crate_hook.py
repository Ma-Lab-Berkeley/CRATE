import torch
import model.crate as crate
import matplotlib.pyplot as plt
import numpy as np
from vis_utils.coding_rate import CodingRate
from einops import rearrange, repeat
from vis_utils.plot import *
import argparse


coding_rate_list = []
sparsity_list = []
def forward_hook_codingrate(module, input, output):
    coding_rate_list.append(criterion(rearrange(output, 'b n (h d) -> b h n d', h=model.transformer.heads)))


def forward_hook_sparsity(module, input, output):
    sparsity_list.append(cal_sparsity(output.cpu().numpy(), is_sparse=True))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint_path", type=str, default="checkpoint.pth.tar")
    args.add_argument("--input_path", type=str, default="firstbatch.pt")
    args = args.parse_args()
    """
    Remark: The input_path contains augmented image tensors, with expected fprmat:
    {'imgs': torch.Size([batch_size, 3, 224, 224])}
    
    For your convenience, we provide an example inputs from image net here:
    https://drive.google.com/file/d/1LTnbVy4HgfaEIGpGdWlsHCF_WAlNkEkH/view?usp=sharing
    """
    
    criterion = CodingRate()
    model = crate.CRATE_small() # change this if you are not using CRATE_small
    input = torch.load(args.input_path)
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')



    new_state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()

    for layer in model.transformer.layers:
        # print(layer[0].fn.qkv)
        layer[0].fn.qkv.register_forward_hook(forward_hook_codingrate)
        layer[1].register_forward_hook(forward_hook_sparsity)
    with torch.no_grad():
        output = model(input['imgs'].cuda())

    means = []
    std_devs = []
    for (mean, std) in coding_rate_list:
        means.append(mean.item())
        std_devs.append(std.item())

    sparsities = []
    std_sparsities = []
    for (mean, std) in sparsity_list:
        sparsities.append(mean)
        std_sparsities.append(std)


    means = [means]
    std_devs = [std_devs]
    sparsities = [sparsities]
    std_sparsities = [std_sparsities]

    plot_coding_rate(means, std_devs)
    plot_sparsity(sparsities, std_sparsities)



