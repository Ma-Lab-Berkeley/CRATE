"""
Code inspired by https://github.com/ShirAmir/dino-vit-features/blob/main/pca.py
"""
import argparse
import PIL.Image
import numpy
import torch
import torch.nn as nn
from pathlib import Path
from vis_utils.shared_extractor import CRATEExtractor
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from typing import List, Tuple
from scipy import ndimage

@torch.no_grad()
def pca(image_paths, load_size: int = 224, layer: int = 11, facet: str = 'key', bin: bool = False, stride: int = 4,
        model_type: str = 'dino_vits8', model: nn.Module = None, n_components: int = 4,
        all_together: bool = True, masks = None) -> List[Tuple[Image.Image, numpy.ndarray]]:
    """
    finding pca of a set of images.
    :param image_paths: a list of paths of all the images.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :param n_components: number of pca components to produce.
    :param all_together: if true apply pca on all images together.
    :return: a list of lists containing an image and its principal components.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    extractor = CRATEExtractor(model_type, stride, device=device, model=model)
    descriptors_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []
    mask_indices_list = []
    # extract descriptors and saliency maps for each image
    img_idx = 0
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        if masks is not None:
            mask = masks[img_idx]
            img_idx += 1
            #compute 1d mask indices
            mask_indices = np.where(mask.flatten())[0]
            # print("mask_indices", mask_indices)
            mask_indices_list.append(mask_indices)
        image_pil_list.append(image_pil)
        descs_all = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls=False).cpu().numpy()
        if masks is not None:
            descs = [descs_all[:, :, i, :] for i in mask_indices]
            descs = np.concatenate(descs, axis=2).reshape(1, 1, -1, descs_all.shape[-1])
        else:
            descs = descs_all
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        descriptors_list.append(descs)
    if all_together:
        descriptors = np.concatenate(descriptors_list, axis=2)[0, 0]
        pca = PCA(n_components=n_components).fit(descriptors)
        pca_descriptors = pca.transform(descriptors)
        if masks is not None:
            split_idxs = np.array([len(indices) for indices in mask_indices_list])
        else:
            split_idxs = np.array([num_patches[0] * num_patches[1] for num_patches in num_patches_list])
        split_idxs = np.cumsum(split_idxs)
        pca_per_image_temp = np.split(pca_descriptors, split_idxs[:-1], axis=0)
        pca_per_image = []
        if masks is not None:
            for i, pcas in enumerate(pca_per_image_temp):
                real_pcas = np.zeros((num_patches_list[i][0] * num_patches_list[i][1], n_components))
                # print()
                real_pcas[mask_indices_list[i], :] = pcas
                # assign min value for masked indices
                real_pcas[real_pcas == 0] = real_pcas.min() - 1
                pca_per_image.append(real_pcas)
        else:
            pca_per_image = pca_per_image_temp
    else:
        pca_per_image = []
        for descriptors in descriptors_list:
            pca = PCA(n_components=n_components).fit(descriptors[0, 0])
            pca_descriptors = pca.transform(descriptors[0, 0])
            pca_per_image.append(pca_descriptors)
    results = [(pil_image, img_pca.reshape((num_patches[0], num_patches[1], n_components))) for
               (pil_image, img_pca, num_patches) in zip(image_pil_list, pca_per_image, num_patches_list)]
    return results


def plot_pca(pil_image: Image.Image, pca_image: numpy.ndarray, last_components_rgb: bool = True,
             save_resized=True,):
    """
    finding pca of a set of images.
    :param pil_image: The original PIL image.
    :param pca_image: A numpy tensor containing pca components of the image. HxWxn_components
    :param save_dir: if None than show results.
    :param last_components_rgb: If true save last 3 components as RGB image in addition to each component separately.
    :param save_resized: If true save PCA components resized to original resolution.
    :param save_prefix: optional. prefix to saving
    :return: a list of lists containing an image and its principal components.
    """
    # save_dir = Path(save_dir)
    # save_dir.mkdir(exist_ok=True, parents=True)
    # pil_image_path = save_dir / f'{save_prefix}_orig_img.png'
    # pil_image.save(pil_image_path)

    pca_images = []
    n_components = pca_image.shape[2]
    for comp_idx in range(n_components):
        comp = pca_image[:, :, comp_idx]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        # comp_file_path = save_dir / f'{save_prefix}_{comp_idx}.png'
        pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
        if save_resized:
            pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
        # pca_pil.save(comp_file_path)
        pca_images.append(pca_pil)

    if last_components_rgb:
        # comp_idxs = f"{n_components-3}_{n_components-2}_{n_components-1}"
        comp = pca_image[:, :, -3:]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        # comp_file_path = save_dir / f'{save_prefix}_{comp_idxs}_rgb.png'
        pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
        if save_resized:
            pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
        # pca_pil.save(comp_file_path)
        pca_images.append(pca_pil)
    return pil_image, pca_images
        
            

def plot_pca_mask(pil_image: Image.Image, pca_image: numpy.ndarray, save_resized=True, th = 0.5):
    """
    finding pca of a set of images.
    :param pil_image: The original PIL image.
    :param pca_image: A numpy tensor containing pca components of the image. HxWxn_components
    :param save_dir: if None than show results.
    :param last_components_rgb: If true save last 3 components as RGB image in addition to each component separately.
    :param save_resized: If true save PCA components resized to original resolution.
    :param save_prefix: optional. prefix to saving
    :return: a list of lists containing an image and its principal components.
    """
    # save_dir = Path(save_dir)
    # save_dir.mkdir(exist_ok=True, parents=True)
    # pil_image_path = save_dir / f'{save_prefix}_orig_img.png'
    # pil_image.save(pil_image_path)

    n_components = pca_image.shape[2]
    # for threshold in [0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
    for threshold in [th]:
        for comp_idx in range(n_components):
            comp = pca_image[:, :, comp_idx]
            comp_min = comp.min(axis=(0, 1))
            comp_max = comp.max(axis=(0, 1))
            comp_img = (comp - comp_min) / (comp_max - comp_min)
            
            comp_img = comp_img > threshold
            # pseudo_mask = densecrf(np.array(I_new), bipartition)
            
            comp_img = ndimage.binary_fill_holes(comp_img>=0.5)
            # print(comp_img.shape)
            # comp_file_path = save_dir / f'{save_prefix}_{comp_idx}_{threshold}.png'
            pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
            if save_resized:
                pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
                # return to tensor
                # comp_img = np.array(pca_pil)
            # pca_pil.save(comp_file_path)
    
    return comp_img

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')