"""Train the leaf-disease classifier. Same flags as train_pest.py; output $AGROARMOUR_WORK/runs/disease/<name>/.

  python ml/train/train_disease.py ml/configs/train_disease_n224.yaml --name D1
"""
from train_pest import main

if __name__ == "__main__":
    main("disease", key="metrics/accuracy_top1", key2="metrics/accuracy_top5", data_tag="disease27 v2")
