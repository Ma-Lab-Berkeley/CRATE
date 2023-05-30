# CRATE :takeout_box:
Code for CRATE :takeout_box:.

## finetune on CIFAR10, CIFAR100, Oxford Flower, Oxford Pet
```python
python finetune.py --bs BATCH_SIZE --net [rit_tiny, rit_small, rit_base, rit_large, vit_tiny, vit_small] --opt [adamW, adam, sgd] --lr LEARNING_RATE --n_epochs 200 --randomaug 1 --data [cifar10, cifar100, pets, flower] --type MODEL_SCALE4RIT --ckpt_dir CKPT_DIR --data_dir DATA_DIR
```