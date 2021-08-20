#!/bin/bash
#SBATCH -J zeroshot1
#SBATCH -o zeroshot1.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH --gres=gpu:1              # 单个节点使用 1 块 GPU 卡
#SBATCH --cpus-per-task=8         # 单任务使用的 CPU 核心数为 4

# zero shot test on semi-supervised pretrained model
python -u train.py --model maskedlm --data fewnerd --prompt hard3  --test_only --model_name distant-pretraining/result/bert-base-cased-2e-05/5000

# zero shot test on bert-base-cased
#python -u train.py --model maskedlm --data fewnerd --prompt hard3  --test_only --model_name bert-base-cased