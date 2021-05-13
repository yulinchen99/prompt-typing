import argparse
from dataloader import get_loader, get_tokenizer, OpenNERDataset, Sample
from model import MaskedModel
from util import load_tag_mapping, get_tag2inputid, ResultLog
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from tqdm import tqdm
import random
import warnings
from transformers import get_linear_schedule_with_warmup
#from memory_profiler import profile

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

# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def main():
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--tag_list_file', type=str, default='tag_list_coarse.txt')
    parser.add_argument('--train_file', type=str, default='../new-NER-discovery/model/data/data/mydata/train-supervised.txt')
    parser.add_argument('--val_file', type=str, default='../new-NER-discovery/model/data/data/mydata/val-supervised.txt')
    parser.add_argument('--test_file', type=str, default='../new-NER-discovery/model/data/data/mydata/test-supervised.txt')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_step_size', type=int, default=200)
    parser.add_argument('--grad_accum_step', type=int, default=10)
    parser.add_argument('--warmup_step', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='checkpoint')

    args = parser.parse_args()
    # set random seed
    set_seed(args.seed)
    # model checkpoint saving path
    import os
    import datetime
    train_file = args.train_file.split('/')[-1]
    model_save_dir = os.path.join(args.save_dir, f'{args.model_name}-{args.tag_list_file}-{train_file}-seed_{args.seed}')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    args.model_save_dir = model_save_dir

    # result log saving path
    result_save_dir = 'result/'
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    result_save_path = os.path.join(result_save_dir, now+'.json')
    resultlog = ResultLog(args, result_save_path)
    

    # get tag list
    print('get tag list...')
    tag_mapping = load_tag_mapping(args.tag_list_file)
    tag_list = list(set(tag_mapping.values()))
    out_dim = len(tag_list)
    tag2idx = {tag:idx for idx, tag in enumerate(tag_list)}
    idx2tag = {idx:tag for idx, tag in enumerate(tag_list)}

    # initialize tokenizer and model
    print('initializing tokenizer and model...')
    tokenizer = get_tokenizer(args.model_name)
    tag2inputid = get_tag2inputid(tokenizer, tag_list)
    model = MaskedModel(args.model_name, idx2tag, tag2inputid, out_dim=out_dim).cuda()

    # initialize dataloader
    print('initializing data...')
    train_dataloader = get_loader(args.train_file, tokenizer, args.batch_size, args.max_length, args.sample_num, tag2idx, tag_mapping)
    val_dataloader = get_loader(args.val_file, tokenizer, args.val_batch_size, args.max_length, args.sample_num, tag2idx, tag_mapping)
    test_dataloader = get_loader(args.test_file, tokenizer, args.val_batch_size, args.max_length, args.sample_num, tag2idx, tag_mapping)

    Loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    global_train_iter = int(args.epoch * len(train_dataloader) / args.grad_accum_step) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=global_train_iter)

    # train
    print('######### start training ##########')
    epoch = args.epoch
    step = 0
    for i in range(epoch):
        print(f'---------epoch {i}---------')
        model.train()
        step_loss = []
        step_acc = []
        # result for each epoch
        result_data = {}
        epoch_acc = []
        epoch_loss = []
        epoch_val_acc = []

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
                print()
                print('[TRAIN STEP %d] loss: %.4f, accuracy: %.4f%%' % (step, np.mean(step_loss), np.mean(step_acc)*100))
                print()

                epoch_acc.append(np.mean(step_acc))
                epoch_loss.append(np.mean(step_loss))

                step_loss = []
                step_acc = []
                torch.cuda.empty_cache()

        result_data['train_acc'] = np.mean(epoch_acc)
        result_data['train_loss'] = np.mean(epoch_loss)

        # validation
        print('########### start validating ##########')
        with torch.no_grad():
            model.eval()
            for data in tqdm(val_dataloader):
                to_cuda(data)
                _, tag_score = model(data)
                tag_pred = torch.argmax(tag_score, dim=1)
                acc = accuracy_score(data['tag_labels'].cpu().numpy(), tag_pred.cpu().numpy())
                epoch_val_acc.append(acc)
            print('[EPOCH %d EVAL RESULT] accuracy: %.4f%%' % (i, np.mean(epoch_acc)*100))
            result_data['val_acc'] = np.mean(epoch_val_acc)
            torch.save(model, args.model_save_dir + f'/checkpoint-{i}')
            print('checkpoint saved')
            print()

            # save training result
            resultlog.update(i, result_data)
            print('result log saved')

if __name__ == '__main__':
    main()


        






