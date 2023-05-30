# CRATE :takeout_box:
Code for CRATE :takeout_box:.


## Finetuning pretrained CRATE

| data | optimizer | lr | n_epochs | bs |
| -------- | -------- | -------- | -------- | -------- |
| cifar10    | adamW   | 5e-5   | 200 | 256 |
| cifar100    | adamW   | 5e-5   | 200 | 256 |
| pets    | adamW   | 1e-4   | 400 | 256 |
| flower | adamW | 1e-4 | 400 | 256 |

```python
python finetune.py --bs BATCH_SIZE --net [CRATE_tiny, CRATE_small, CRATE_base, CRATE_large, vit_tiny, vit_small] --opt [adamW, adam, sgd] --lr LEARNING_RATE --n_epochs N_EPOCHS --randomaug 1 --data [cifar10, cifar100, pets, flower] --type MODEL_SCALE4CRATE --ckpt_dir CKPT_DIR --data_dir DATA_DIR
```