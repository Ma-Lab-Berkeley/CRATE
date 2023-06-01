# CRATE (Coding RAte reduction TransformEr)
This repository is the official PyTorch implementation of the paper [White-Box Transformers via Sparse Rate Reduction](https://arxiv.org/abs/2306.xxxxx) (2023) 

by [Yaodong Yu](https://yaodongyu.github.io) (UC Berkeley), [Sam Buchanan](https://sdbuchanan.com) (TTIC), [Druv Pai](https://druvpai.github.io) (UC Berkeley), [Tianzhe Chu](https://tianzhechu.com/) (UC Berkeley), [Ziyang Wu](https://robinwu218.github.io/) (UC Berkeley), [Shengbang Tong](https://tsb0601.github.io/petertongsb/) (UC Berkeley), [Benjamin D Haeffele](https://www.cis.jhu.edu/~haeffele/#about) (Johns Hopkins University), and [Yi Ma](http://people.eecs.berkeley.edu/~yima/) (UC Berkeley). 


## What is CRATE?
CRATE (Coding RAte reduction TransformEr) is a white-box (mathematically interpretable) transformer architecture, where each layer performs a single step of an alternating minimization algorithm to optimize the **sparse rate reduction objective**
 <p align="center">
    <img src="figs/fig_objective.png" width="800"\>
</p>
<p align="center">

where the $\ell^{0}$-norm promotes the sparsity of the final token representations $\mathbf{Z} = f(\mathbf{X})$. The function $f$ is defined as 
$$f=f^{L} \circ f^{L-1} \circ \cdots \circ f^{1} \circ f^{0},$$
$f^0$ is the pre-processing mapping, and $f^{\ell}$ is the $\ell$-th layer forward mapping that transforms the token distribution to optimize the above sparse rate reduction objective incrementally. More specifically, $f^{\ell}$ transforms the $\ell$-th layer token representations $\mathbf{Z}^{\ell}$ to  $\mathbf{Z}^{\ell+1}$ via the $\texttt{MSSA}$ (Multi-Head Subspace Self-Attention) block and the $\texttt{ISTA}$ (Iterative Shrinkage-Thresholding Algorithms) block, i.e.,
$$\mathbf{Z}^{\ell+1} = f^{\ell}(\mathbf{Z}^{\ell}) = \texttt{ISTA}(\texttt{MSSA}(\mathbf{Z}^{\ell})).$$
Figure 1 presents an overview of the pipeline for our proposed **CRATE** architecture:

<p align="center">
    <img src="figs/fig1.png" width="800"\>
</p>
<p align="center">


Figure 2 shows the overall architecture of one block of **CRATE**:

<p align="center">
    <img src="figs/fig_arch.png" width="800"\>
</p>
<p align="center">

In Figure 3, we measure the compression term [ $R^{c}$ ($\mathbf{Z}^{\ell+1/2}$) ] and the sparsity term [ $||\mathbf{Z}^{\ell+1}||_0$ ] defined in the **sparse rate reduction objective**, and we find that each layer of **CRATE** indeed optimizes the targeted objectives:
<p align="center">
    <img src="figs/fig3.png" width="900"\>
</p>
<p align="center">


## Construct a CRATE model
A CRATE model can be defined using the following code. The given parameters are specified for CRATE-tiny.
```python
from model.crate import CRATE
net = CRATE(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=384,
    depth=12,
    heads=6,
    mlp_dim=384,
    dropout=0.0,
    emb_dropout=0.0,
    dim_head=384//6
    )
```
## Training CRATE on ImageNet
```python
python main.py --arch [CRATE_tiny, CRATE_small, CRATE_base, CRATE_large, vit_tiny, vit_small] --batch-size BATCH_SIZE --epochs EPOCHS --optimizer Lion --lr LEARNING_RATE --weight-decay WEIGHT_DECAY --print-freq 25 --data DATA_DIR
```

As an example, we use the following command for training CRATE_tiny:
```python
python main.py python main.py --arch CRATE_tiny --batch-size 512 --epochs 200 --optimizer Lion --lr 0.0002 --weight-decay 0.04  --print-freq 25 --data DATA_DIR
```

## Finetuning pretrained CRATE

| data | optimizer | lr | n_epochs | bs |
| -------- | -------- | -------- | -------- | -------- |
| cifar10    | adamW   | 5e-5   | 200 | 256 |
| cifar100    | adamW   | 5e-5   | 200 | 256 |
| pets    | adamW   | 1e-4   | 400 | 256 |
| flower | adamW | 1e-4 | 400 | 256 |

```python
python finetune.py --bs BATCH_SIZE --net [CRATE_tiny, CRATE_small, CRATE_base, CRATE_large, vit_tiny, vit_small] 
    --opt [adamW, adam, sgd] --lr LEARNING_RATE --n_epochs N_EPOCHS --randomaug 1 --data [cifar10, cifar100, pets, flower] 
    --type MODEL_SCALE4CRATE --ckpt_dir CKPT_DIR --data_dir DATA_DIR
```
