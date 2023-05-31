# CRATE :takeout_box:
Code for CRATE :takeout_box:.

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
    --opt [adamW, adam, sgd] --lr LEARNING_RATE --n_epochs N_EPOCHS --randomaug 1 --data [cifar10, cifar100, pets, flower] --type MODEL_SCALE4CRATE --ckpt_dir CKPT_DIR --data_dir DATA_DIR
```