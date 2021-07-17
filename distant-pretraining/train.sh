#!/bin/bash
#SBATCH -J test
#SBATCH -o test-bert.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH --gres=gpu:2              # 单个节点使用 1 块 GPU 卡
#SBATCH --cpus-per-task=8         # 单任务使用的 CPU 核心数为 4
#5e-4, 1e-3
python -u train.py --model_name bert-base-cased --datapath ./data/samples.json --eval_step 5000 --lr 2e-5 --train_batch_size 16 --epoch 20 --train_print_step 500 --grad_accum_step 1 --train_batch_size 16
