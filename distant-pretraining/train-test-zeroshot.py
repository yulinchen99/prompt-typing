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
parser.add_argument('--ckpt_name', type=str, default=None)

args = parser.parse_args()

device='cuda:0'
model_name = args.model_name.split('/')[-1]
label_name = args.labelpath.split('/')[-1]
model_save_path = f'/mnt/sfs_turbo/cyl/ner-mlm/distant-pretraining/result/{model_name}-{label_name}-{args.lr}'
if args.ckpt_name:
    model_save_path += f'-{args.ckpt_name}'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
with open(model_save_path + '/report.txt', 'a+')as f:
    f.writelines(str(args))

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_prior_distribution(model, datalist, sample_num=1000):
    model.eval()
    calib_data = random.sample(datalist, sample_num)
    calib_dataloader = get_loader(calib_data, 200)
    prior_dist = []
    with torch.no_grad():
        for data, label in tqdm(calib_dataloader):
            label = label.to(device)
            tag_dist = model.get_prior_distribution(data)
            prior_dist.append(tag_dist)
        prior_dist = torch.cat(prior_dist, dim=0)
    prior_dist = torch.mean(prior_dist, dim=0)
    return prior_dist


def evaluate(model, step, test_dataloader, Loss, prior_dist=None):
    print('start testing...')
    model.eval()
    test_loss = []
    test_acc = []
    with torch.no_grad():
        for data, label in tqdm(test_dataloader):
            label = label.to(device)
            sent1_embed, sent2_embed, score = model(data, prior_dist = prior_dist)
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
    datalist = random.sample(datalist, 1000000)
    train, test, _, _ = train_test_split(datalist, [0]*len(datalist), random_state=0, test_size=0.01)
    print('building dataloader...')
    train_dataloader = get_loader(train, args.train_batch_size)
    test_dataloader = get_loader(test, args.test_batch_size)


    # get label_ids
    label_ids = None
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

    prior_dist = None
    test_prior_dist = None

    # print('get prior distribution')
    # prior_dist = get_prior_distribution(model, datalist)


    # train_loss = []
    # test_loss = []
    step = 0

    # save, load and test
    save_path = model_save_path + f'/{step}'
    model.save(save_path)
    os.system(f'python -u /mnt/sfs_turbo/cyl/ner-mlm/train.py --model maskedlm --data {label_name} --prompt hard3  --test_only --model_name {save_path} --result_save_dir self-zeroshot-result')

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
            sent1_embed, sent2_embed, score = model(data, prior_dist = prior_dist)
            #print(score)
            pred = (score + 0.5).floor()
            acc = accuracy_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy())

            loss = Loss(sent1_embed, sent2_embed, score, label)
            loss.backward()

            train_step_acc.append(acc)
            train_report_loss.append(loss.item())
            #except:
                #print(f'ERROR on step {step}:', sent1_embed, data, list(model.named_parameters()))
                #step += 1
                #continue

            if (step+1) % args.grad_accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step+1) % (args.train_print_step) == 0:
                print('[TRAIN STEP %d] loss: %.4f, acc: %.4f' % (step, np.mean(train_report_loss), np.mean(train_step_acc)))
                train_step_loss += train_report_loss
                train_report_loss = []
                train_step_acc = []
                torch.cuda.empty_cache()

                # update prior distribution
                # print('update prior distribution')
                # prior_dist = get_prior_distribution(model, datalist)
                model.train()

            if (step+1) % args.eval_step == 0:
                save_path = model_save_path + f'/{step}'
                model.save(save_path)
                os.system(f'python -u /mnt/sfs_turbo/cyl/ner-mlm/train.py --model maskedlm --data {label_name} --prompt hard3  --test_only --model_name {save_path} --result_save_dir {label_name}-zeroshot-result')
                # train_loss.append(np.mean(train_step_loss))

                # print('get prior distribution')
                # test_prior_dist = get_prior_distribution(model, test)

                # loss = evaluate(model, step, test_dataloader, Loss, prior_dist = test_prior_dist)

                # test_loss.append(loss)

                model.train()
            step += 1


    # np.save(model_save_path + '/train_loss.npy', np.array(train_loss))
    # np.save(model_save_path + '/test_loss.npy', np.array(test_loss))


if __name__ == '__main__':
    # os.system(f'python -u /mnt/sfs_turbo/cyl/ner-mlm/train.py --model maskedlm --data {label_name} --prompt hard3  --test_only --model_name /mnt/sfs_turbo/cyl/ner-mlm/bert --result_save_dir self-zeroshot-result')
    train()


