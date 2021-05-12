#!/bin/bash
#SBATCH -o train.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH --gres=gpu:2              # 单个节点使用 1 块 GPU 卡
#SBATCH --cpus-per-task=8         # 单任务使用的 CPU 核心数为 4

sh train.sh