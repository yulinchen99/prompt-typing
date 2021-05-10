import argparse
from dataloader import get_loader, get_tokenizer, OpenNERDataset, Sample
from model import MaskedModel
from util import load_tag_list
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from tqdm import tqdm
import random
import warnings
from memory_profiler import profile

warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def to_cuda(data):
    for item in data:
        data[item] = data[item].cuda()

@profile(precision=4,stream=open('memory_profiler.log','w+'))
def main():
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--tag_list_file', type=str, default='tag_list.txt')
    parser.add_argument('--train_file', type=str, default='../model/data/data/mydata/train-inter-new.txt')
    parser.add_argument('--val_file', type=str, default='../model/data/data/mydata/val-inter-new.txt')
    parser.add_argument('--test_file', type=str, default='../model/data/data/mydata/test-inter-new.txt')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_step_size', type=int, default=200)
    parser.add_argument('--grad_accum_step', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoint')

    args = parser.parse_args()

    set_seed(args.seed)

    # get tag list
    print('get tag list...')
    tag_list = load_tag_list(args.tag_list_file)
    out_dim = len(tag_list)
    # initialize tokenizer and model
    print('initializing model...')
    tokenizer = get_tokenizer(args.model_name)
    model = MaskedModel(args.model_name, out_dim).cuda()

    Loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size)

    # initialize dataloader
    print('initializing data...')
    train_dataloader = get_loader(args.train_file, tokenizer, args.batch_size, args.max_length, args.sample_num, tag_list)
    val_dataloader = get_loader(args.val_file, tokenizer, args.val_batch_size, args.max_length, args.sample_num, tag_list)
    test_dataloader = get_loader(args.test_file, tokenizer, args.val_batch_size, args.max_length, args.sample_num, tag_list)
    # train
    print('######### start training ##########')
    epoch = args.epoch
    step = 0
    for i in range(epoch):
        print(f'---------epoch {i}---------')
        model.train()
        step_loss = []
        step_acc = []
        j = 0

        for data in tqdm(train_dataloader):
            to_cuda(data)
            word_loss, tag_score = model(data)
            loss = Loss(tag_score, data['tag_labels']) + word_loss
            del word_loss
            loss.backward()
            step_loss.append(loss.item())

            tag_pred = torch.argmax(tag_score, dim=1)
            del tag_score
            acc = accuracy_score(data['tag_labels'].cpu().numpy(), tag_pred.cpu().numpy())
            step_acc.append(acc)

            step += 1

            if step % args.grad_accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print('[TRAIN STEP %d] loss: %.4f, accuracy: %.4f%%' % (step, np.mean(step_loss), np.mean(step_acc)*100))
                print()

                step_loss = []
                step_acc = []
                torch.cuda.empty_cache()

        # validation
        print('########### start validating ##########')
        with torch.no_grad():
            model.eval()
            epoch_acc = []
            for data in tqdm(val_dataloader):
                to_cuda(data)
                _, tag_score = model(data)
                tag_pred = torch.argmax(tag_score, dim=1)
                acc = accuracy_score(data['tag_labels'].cpu().numpy(), tag_pred.cpu().numpy())
                epoch_acc.append(acc)
            print('[EPOCH %d EVAL RESULT] accuracy: %.4f%%' % (i, np.mean(epoch_acc)))
            torch.save(model, args.save_dir+f'/{args.model_name}-checkpoint-{i}')
            print('checkpoint saved')
            print()

if __name__ == '__main__':
    main()


        






