from model.crate_ae.crate_ae import mae_crate_base
# from crate_ae_extractor import crate_base

import math
from typing import Union, List, Tuple
import types
import torch.nn.modules.utils as nn_utils
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from pathlib import Path

class CRATEExtractor:
    def __init__(self, model_type: str = 'crate_mae_b16', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):

        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            raise NotImplementedError

        self.model = CRATEExtractor.patch_vit_resolution(self.model, stride=stride, model_type = model_type)
        self.model.eval()
        self.model.to(self.device)
        if model_type == 'crate_mae_b16':
            self.p = self.model.patch_embed.patch_size[0]
            self.stride = self.model.patch_embed.proj.stride
        else:
            raise NotImplementedError
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225) 

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:

            npatch = x.shape[1] - 1
            N = self.decoder_pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.decoder_pos_embed
            class_pos_embed = self.decoder_pos_embed[:, 0]
            patch_pos_embed = self.decoder_pos_embed[:, 1:]
            
            dim = x.shape[-1]
            # dim = self.dim
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            # print(w0, h0, npatch)
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            # print("patch_pos_shape:", patch_pos_embed.shape)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            # print("patch_pos_shape:", torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).shape)
            # print(torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1))
            # exit()
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int, model_type: str) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        if model_type == 'crate_mae_b16':
            patch_size = model.patch_embed.patch_size[0]
        else:
            raise NotImplementedError
        # if stride == patch_size:  # nothing to do
        #     return model

        stride = nn_utils._pair(stride)
        # print(patch_size)
        # print(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        # model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(CRATEExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize((load_size, load_size), interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image



    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        
        B, C, H, W = batch.shape
        # print("batch shape:", batch.shape)
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        
        w = self.num_patches[1]
        h = self.num_patches[0]
        # shaper = torch.zeros((1, w * h + 1, self.model.dim))
        
        # self.model.interpolated_pos_embed= nn.Parameter(self.model.interpolate_pos_encoding(shaper, W, H))
        qkv = self.model.get_last_key_enc(batch, layer = layer)
        qkv = qkv[None, :, :, :]
        # qkv = (qkv.reshape(bs, nb_token, 1, nb_head, -1).permute(2, 0, 3, 1, 4))
        # print("step:", qkv.shape)
        # qkv = qkv[0]
        # print("qkv.shape", qkv.shape)
        # k = qkv.transpose(1,2).reshape(bs, nb_token, -1)
        # feats = k[:, 1:].transpose(1,2).reshape(bs, self.feat_dim, feat_h * feat_w)
        return qkv[:, :, 1:, :]