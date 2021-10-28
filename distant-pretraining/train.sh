#!/bin/bash
#SBATCH -J test
#SBATCH -o test-bert.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH --gres=gpu:2              # 单个节点使用 1 块 GPU 卡
#SBATCH --cpus-per-task=8         # 单任务使用的 CPU 核心数为 4
#5e-4, 1e-3
python -u /mnt/sfs_turbo/cyl/ner-mlm/distant-pretraining/train-test-zeroshot.py --model_name /mnt/sfs_turbo/cyl/ner-mlm/bert --datapath /mnt/sfs_turbo/cyl/ner-mlm/distant-pretraining/data/samples.json --labelpath /mnt/sfs_turbo/cyl/ner-mlm/data/fewnerd --eval_step 2000 --lr 2e-5 --train_batch_size 128 --epoch 3 --train_print_step 100 --grad_accum_step 1 --test_batch_size 200 --ckpt_name wsdiv
