#!/bin/bash
#SBATCH -J onto
#SBATCH -o train-onto.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH --gres=gpu:2              # 单个节点使用 1 块 GPU 卡
#SBATCH --cpus-per-task=8         # 单任务使用的 CPU 核心数为 4

python -u train-onto.py --model_name roberta-base --tag_list_file OntoNotes-89/train_clean.tsv --train_file OntoNotes-89/train_clean.tsv --val_file OntoNotes-89/dev.txt --test_file OntoNotes-89/test_clean.tsv