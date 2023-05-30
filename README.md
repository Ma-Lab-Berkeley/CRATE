# CRATE :takeout_box:
Code for CRATE :takeout_box:.


## finetune on CIFAR10, CIFAR100, Oxford Flower, Oxford Pet
| Column 1 | Column 2 | Column 3 |
| -------- | -------- | -------- |
| Row 1    | Data 1   | Data 2   |
| Row 2    | Data 3   | Data 4   |
| Row 3    | Data 5   | Data 6   |

```python
python finetune.py --bs BATCH_SIZE --net [rit_tiny, rit_small, rit_base, rit_large, vit_tiny, vit_small] --opt [adamW, adam, sgd] --lr LEARNING_RATE --n_epochs N_EPOCHS --randomaug 1 --data [cifar10, cifar100, pets, flower] --type MODEL_SCALE4RIT --ckpt_dir CKPT_DIR --data_dir DATA_DIR
```