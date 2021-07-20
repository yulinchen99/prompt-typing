import sys
import os
from util.data import get_loader, load_data
from util.model import PretrainModel, MTBLoss
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score
from util.util import get_tokenizer, load_tag_mapping, get_label_ids

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_name', type=str, default='roberta-base', help='bert, roberta, and gpt2 are supported')
#parser.add_argument('--max_length', type=int, default=64)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--datapath', type=str, default='./data/samples.json')
parser.add_argument('--labelpath', type=str, default='../data/fewnerd')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_step_size', type=int, default=200)
parser.add_argument('--train_print_step', type=int, default=10)
parser.add_argument('--grad_accum_step', type=int, default=10)
parser.add_argument('--warmup_step', type=int, default=100)
parser.add_argument('--eval_step', type=int, default=100, help='val every x steps of training')
#parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--load_ckpt', type=str, default=None)
#parser.add_argument('--ckpt_name', type=str, default=None)

args = parser.parse_args()

device='cuda:0'
model_save_path = f'./result/{args.model_name}-{args.lr}'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(model, step, test_dataloader, Loss):
    print('start testing...')
    model.eval()
    test_loss = []
    test_acc = []
    with torch.no_grad():
        for data, label in tqdm(test_dataloader):
            label = label.to(device)
            sent1_embed, sent2_embed, score = model(data)
            pred = (score + 0.5).floor()
            acc = accuracy_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy())
            loss = Loss(sent1_embed, sent2_embed, score, label)

            test_loss.append(loss.item())
            test_acc.append(acc)

    print('STEP %d: test loss %.4f, test acc: %.4f\n'%(step, np.mean(test_loss), np.mean(test_acc)))
    with open(model_save_path + '/report.txt', 'a+')as f:
        f.writelines('STEP %d: test loss %.4f, test acc: %.4f\n'%(step, np.mean(test_loss), np.mean(test_acc)))
    save_path = model_save_path + f'/{step}'
    model.save(save_path)
    return np.mean(test_loss)


def train():
    set_seed(seed=args.seed)
    # load data
    print('loading data...')
    datalist = load_data(args.datapath)
    #datalist = random.sample(datalist, 1000000)
    train, test, _, _ = train_test_split(datalist, [0]*len(datalist), random_state=0, test_size=0.1)
    print('building dataloader...')
    train_dataloader = get_loader(train, args.train_batch_size)
    test_dataloader = get_loader(test, args.test_batch_size)

    # get label_ids
    tokenizer = get_tokenizer(args.model_name)
    tag_mapping = load_tag_mapping(args.labelpath)
    label_ids = get_label_ids(tokenizer, tag_mapping)
    #print(label_ids)


    # initialize model 
    print('initializing model...')
    model = PretrainModel(args.model_name, device=device, label_ids=label_ids)
    if args.load_ckpt:
        model.load(args.load_ckpt)
    if 'cuda' in device:
        model = model.cuda()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    global_train_iter = int(args.epoch * len(train_dataloader) / args.grad_accum_step + 0.5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=global_train_iter)

    # loss
    Loss = MTBLoss()

    train_loss = []
    test_loss = []
    step = 0
    if args.load_ckpt:
        step = int(args.load_ckpt.split('/')[-1])
    train_report_loss = []
    train_step_loss = []
    train_step_acc = []

    print('start training...')
    for i in range(args.epoch):
        print(f'-----------epoch {i}----------------')
        for data, label in tqdm(train_dataloader):
            #try:
            label = label.to(device)
            sent1_embed, sent2_embed, score = model(data)
            #print(score)
            pred = (score + 0.5).floor()
            acc = accuracy_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy())

            loss = Loss(sent1_embed, sent2_embed, score, label)
            loss.backward()

            train_step_acc.append(acc)
            train_report_loss.append(loss.item())
            step += 1
            #except:
                #print(f'ERROR on step {step}:', sent1_embed, data, list(model.named_parameters()))
                #step += 1
                #continue

            if step % args.grad_accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % (args.train_print_step) == 0:
                print('[TRAIN STEP %d] loss: %.4f, acc: %.4f' % (step, np.mean(train_report_loss), np.mean(train_step_acc)))
                train_step_loss += train_report_loss
                train_report_loss = []
                train_step_acc = []
                torch.cuda.empty_cache()

            if step % args.eval_step == 0:
                train_loss.append(np.mean(train_step_loss))
                loss = evaluate(model, step, test_dataloader, Loss)
                test_loss.append(loss)
                model.train()

    np.save(model_save_path + '/train_loss.npy', np.array(train_loss))
    np.save(model_save_path + '/test_loss.npy', np.array(test_loss))


if __name__ == '__main__':
    train()

