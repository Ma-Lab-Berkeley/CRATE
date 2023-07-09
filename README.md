# CRATE (Coding RAte reduction TransformEr)
This repository is the official PyTorch implementation of the paper [White-Box Transformers via Sparse Rate Reduction](https://arxiv.org/abs/2306.01129) (2023) 

by [Yaodong Yu](https://yaodongyu.github.io) (UC Berkeley), [Sam Buchanan](https://sdbuchanan.com) (TTIC), [Druv Pai](https://druvpai.github.io) (UC Berkeley), [Tianzhe Chu](https://tianzhechu.com/) (UC Berkeley), [Ziyang Wu](https://robinwu218.github.io/) (UC Berkeley), [Shengbang Tong](https://tsb0601.github.io/petertongsb/) (UC Berkeley), [Benjamin D Haeffele](https://www.cis.jhu.edu/~haeffele/#about) (Johns Hopkins University), and [Yi Ma](http://people.eecs.berkeley.edu/~yima/) (UC Berkeley). 


## What is CRATE?
CRATE (Coding RAte reduction TransformEr) is a white-box (mathematically interpretable) transformer architecture, where each layer performs a single step of an alternating minimization algorithm to optimize the **sparse rate reduction objective**
 <p align="center">
    <img src="figs/fig_objective.png" width="700"\>
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
A CRATE model can be defined using the following code, (the below parameters are specified for CRATE-Tiny)
```python
from model.crate import CRATE
dim = 384
n_heads = 6
depth = 12
model = CRATE(image_size=224,
              patch_size=16,
              num_classes=1000,
              dim=dim,
              depth=depth,
              heads=n_heads,
              dim_head=dim // n_heads)
```

### Pre-trained Checkpoints (ImageNet-1K)
| model | `dim` | `n_heads` | `depth` | pre-trained checkpoint |
| -------- | -------- | -------- | -------- | -------- | 
| **CRATE-T**(tiny)    | 384   | 6   | 12 | TODO | 
| **CRATE-S**(small)    | 576   | 12   | 12 | [download link](https://drive.google.com/file/d/1hYgDJl4EKHYfKprwhEjmWmWHuxnK6_h8/view?usp=share_link) | 
| **CRATE-B**(base)    | 768   | 12   | 12 | TODO | 
| **CRATE-L**(large) | 1024 | 16 | 24 | TODO | 

## Training CRATE on ImageNet
To train a CRATE model on ImageNet-1K, run the following script (training CRATE-tiny)

As an example, we use the following command for training CRATE-tiny on ImageNet-1K:
```python
python main.py 
  --arch CRATE_tiny 
  --batch-size 512 
  --epochs 200 
  --optimizer Lion 
  --lr 0.0002 
  --weight-decay 0.05 
  --print-freq 25 
  --data DATA_DIR
```
and replace `DATA_DIR` with `[imagenet-folder with train and val folders]`.


## Finetuning pretrained / training random initialized CRATE on CIFAR10

```python
python finetune.py 
  --bs 256 
  --net CRATE_tiny 
  --opt adamW  
  --lr 5e-5 
  --n_epochs 200 
  --randomaug 1 
  --data cifar10 
  --ckpt_dir CKPT_DIR 
  --data_dir DATA_DIR
```
Replace `CKPT_DIR` with the path for the pretrained CRATE weight, and replace `DATA_DIR` with the path for the `CIFAR10` dataset. If `CKPT_DIR` is `None`, then this script is for training CRATE from random initialization on CIFAR10.


## Reference
For technical details and full experimental results, please check the [paper](https://arxiv.org/abs/2306.01129). Please consider citing our work if you find it helpful to yours:

```
@misc{yu2023whitebox,
      title={White-Box Transformers via Sparse Rate Reduction}, 
      author={Yaodong Yu and Sam Buchanan and Druv Pai and Tianzhe Chu and Ziyang Wu and Shengbang Tong and Benjamin D. Haeffele and Yi Ma},
      year={2023},
      eprint={2306.01129},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
