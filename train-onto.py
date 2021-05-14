import argparse
from transformers import BertConfig, RobertaConfig
from dataloader_ontonote import get_loader, get_tokenizer, OpenNERDataset, Sample
from model import MaskedModel
from util import get_tag2inputid, ResultLog
from get_ontonotes_tags import load_tag_mapping
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
    parser.add_argument('--prompt', type=str, default='[P]-[P1]-[P2]')


    args = parser.parse_args()
    # set random seed
    set_seed(args.seed)
    # model checkpoint saving path
    import os
    import datetime
    train_file = args.train_file.split('/')[-1]
    tag_list_file = args.tag_list_file.split('/')[-1]
    model_save_dir = os.path.join(args.save_dir, f'{args.model_name}-{tag_list_file}-{train_file}-{args.prompt}-seed_{args.seed}')
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
    print(len(tag2idx))
    print(tag2idx)

    # initialize tokenizer and model
    print('initializing tokenizer and model...')
    tokenizer = get_tokenizer(args.model_name)
    prompt = args.prompt.split('-')
    added_num = tokenizer.add_tokens(prompt)

    tag2inputid = get_tag2inputid(tokenizer, tag_list)
    vocab_size = tokenizer.vocab_size + added_num
    if 'roberta' in args.model_name:
        config = RobertaConfig(vocab_size=vocab_size)
    elif 'bert' in args.model_name:
        config = BertConfig(vocab_size=vocab_size)
    model = MaskedModel(args.model_name, config, idx2tag, tag2inputid, out_dim=out_dim).cuda()

    # initialize dataloader
    print('initializing data...')
    train_dataloader = get_loader(args.train_file, tokenizer, args.batch_size, args.max_length, args.sample_num, tag2idx, tag_mapping, 4, prompt)
    val_dataloader = get_loader(args.val_file, tokenizer, args.val_batch_size, args.max_length, args.sample_num, tag2idx, tag_mapping, 3, prompt)
    # test_dataloader = get_loader(args.test_file, tokenizer, args.val_batch_size, args.max_length, args.sample_num, tag2idx, tag_mapping, 4, prompt)

    Loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    global_train_iter = int(args.epoch * len(train_dataloader) / args.grad_accum_step) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=global_train_iter)

    # train
    print('######### start training ##########')
    epoch = args.epoch
    step = 0
    result_data = {}
    step_acc = []
    step_loss = []
    step_val_acc = []
    train_step_loss = []
    train_step_acc = []
    for i in range(epoch):
        print(f'---------epoch {i}---------')
        model.train()
        # result for each epoch
        for data in train_dataloader:
            to_cuda(data)
            tag_score = model(data)
            loss = Loss(tag_score, data['tag_labels'])
            loss.backward()
            train_step_loss.append(loss.item())

            tag_pred = torch.argmax(tag_score, dim=1)
            del tag_score
            acc = accuracy_score(data['tag_labels'].cpu().numpy(), tag_pred.cpu().numpy())
            train_step_acc.append(acc)

            step_acc.append(acc)
            step_loss.append(loss.item())

            step += 1

            if step % args.grad_accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % (args.grad_accum_step*10) == 0:
                    print('[TRAIN STEP %d] loss: %.4f, accuracy: %.4f%%' % (step, np.mean(train_step_loss), np.mean(train_step_acc)*100))
                    train_step_loss = []
                    train_step_acc = []
                    torch.cuda.empty_cache()

            # validation
            if step % 2000 == 0:
                print('########### start validating ##########')
                with torch.no_grad():
                    model.eval()
                    for data in tqdm(val_dataloader):
                        to_cuda(data)
                        tag_score = model(data)
                        tag_pred = torch.argmax(tag_score, dim=1)
                        acc = accuracy_score(data['tag_labels'].cpu().numpy(), tag_pred.cpu().numpy())
                        step_val_acc.append(acc)
                    print('[STEP %d EVAL RESULT] accuracy: %.4f%%' % (step, np.mean(step_val_acc)*100))
                    result_data['val_acc'] = np.mean(step_val_acc)
                    torch.save(model, args.model_save_dir + f'/checkpoint-{step}')
                    print('checkpoint saved')
                    print()
                    # save training result
                    result_data['train_acc'] = np.mean(step_acc)
                    result_data['train_loss'] = np.mean(step_loss)
                    resultlog.update(step, result_data)
                    print('result log saved')

                    step_acc = []
                    step_loss = []
                    step_val_acc = []

if __name__ == '__main__':
    main()


        





