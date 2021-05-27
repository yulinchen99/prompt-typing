import argparse
from transformers import BertConfig, RobertaConfig
from util.data_loader import get_loader
from model.baseline import EntityTypingModel as BaselineModel
from model.maskedlm import EntityTypingModel as MaskedLM
from util.util import load_tag_mapping, get_tag2inputid, get_mapped_tag_list, ResultLog, get_tokenizer, PartialLabelLoss
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from tqdm import tqdm
import random
import warnings
from transformers import get_linear_schedule_with_warmup
import os
import datetime
from util.metrics import get_metrics
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
        if isinstance(data[item], torch.LongTensor):
            data[item] = data[item].cuda()

# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def main():
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default='ontonote', help='ontonote, fewnerd or bbn')
    parser.add_argument('--model', type=str, default='maskedlm', help='baseline or maskedlm')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--embed_lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=200)
    parser.add_argument('--grad_accum_step', type=int, default=10)
    parser.add_argument('--warmup_step', type=int, default=100)
    parser.add_argument('--val_step', type=int, default=2000, help='val every x steps of training')
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--load_ckpt', type=str, default=None)

    # for soft prompt only
    parser.add_argument('--prompt', type=str, default='[P]-[P1]-[P2]')
    parser.add_argument('--dual_optim', action='store_true', default=False, help='set True if separate learning rate in maskedlm p-prompt setting is desired')
    parser.add_argument('--dropout', type=float, default=0.1)

    # for baseline only
    parser.add_argument('--usecls', action='store_true', default=False)
    parser.add_argument('--highlight_entity', type=str, default=None, help='for baseline model, highlight tokens around entity')
    parser.add_argument('--loss', type=str, default='cross', help='cross or partial')


    args = parser.parse_args()
    # set random seed
    set_seed(args.seed)

    # data path
    args.data = os.path.join('data', args.data)

    # model saving path
    data = args.data.split('/')[-1]
    MODEL_SAVE_PATH = os.path.join(args.save_dir, f'{args.model}-{args.model_name}-{data}-{args.prompt}-seed_{args.seed}')

    if args.dual_optim and args.model == 'maskedlm':
        MODEL_SAVE_PATH += '-dual_optim'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    args.model_save_path = MODEL_SAVE_PATH
    print('modelsave path:', MODEL_SAVE_PATH)
    
    # prompt
    PROMPT = None
    IS_P_PROMPT = False
    HIGHLIGHT_ENTITY = None
    if args.prompt:
        PROMPT = args.prompt.split('-')
    if '[P]' in PROMPT:
        IS_P_PROMPT = True
    if args.highlight_entity is not None:
        HIGHLIGHT_ENTITY = args.highlight_entity.split('-')

    # check dual_optim
    if args.dual_optim and ((not IS_P_PROMPT) or args.model == 'baseline'):
        print('[ERROR] dual_optim is only supported in MaskedLM and is_p_promt setting')
        raise ValueError    

    # get tag list
    print('get tag list...')
    tag_mapping = load_tag_mapping(args.data)
    mapped_tag_list = get_mapped_tag_list(args.data, tag_mapping)
    out_dim = len(mapped_tag_list)
    tag2idx = {tag:idx for idx, tag in enumerate(mapped_tag_list)}
    idx2tag = {idx:tag for idx, tag in enumerate(mapped_tag_list)}
    print(tag2idx)

    # initialize model
    print('initializing model...')
    if args.model == 'baseline':
        model = BaselineModel(args.model_name, idx2tag, mapped_tag_list, out_dim, highlight_entity=HIGHLIGHT_ENTITY, dropout=args.dropout, usecls=args.usecls)
    elif args.model == 'maskedlm':
        model = MaskedLM(args.model_name, idx2tag, mapped_tag_list, prompt=PROMPT, is_p_prompt=IS_P_PROMPT, dropout=args.dropout)
    else:
        raise NotImplementedError
    model = model.cuda()

    # initialize dataloader
    print(f'initializing data from {args.data}...')
    train_dataloader = get_loader(args.data, 'train', args.batch_size, args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY)
    val_dataloader = get_loader(args.data, 'dev', args.val_batch_size, args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY)
    test_dataloader = get_loader(args.data, 'test', args.val_batch_size, args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY)

    # initialize loss
    if args.loss == 'cross':
        Loss = nn.CrossEntropyLoss()
    elif args.loss == 'partial':
        Loss = PartialLabelLoss()
    else:
        assert False, print(f'invalid loss {args.loss}!')

    # initialize optimizer
    if args.dual_optim:
        print(f'using dual optim, embed_lr: {args.embed_lr}, lr:{args.lr}')
        bert_param = [p for p in model.parameters() if id(p) not in list(map(id, model.prompt_embedding.parameters())) + list(map(id, model.prompt_linear.parameters()))]
        optimizer = AdamW([
                {'params': bert_param},
                {'params': model.prompt_embedding.parameters(), 'lr': args.embed_lr}
            ], lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    global_train_iter = int(args.epoch * len(train_dataloader) / args.grad_accum_step + 0.5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=global_train_iter)

    # result log saving path
    result_save_dir = 'result/'
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    result_save_path = os.path.join(result_save_dir, now+'.json')
    resultlog = ResultLog(args, result_save_path)

    # train
    print('######### start training ##########')
    epoch = args.epoch
    step = 0
    # logging
    result_data = {}
    # info every val step
    step_acc = []
    step_loss = []
    #step_val_acc = []
    # infor every grad accum step
    train_step_loss = []
    train_step_acc = []
    # best acc on val
    best_acc = 0.0

    if not args.test_only:
        if args.load_ckpt is not None:
            print(f'loading pre-trained ckpt {args.load_ckpt}...')
            load_path =  args.load_ckpt
            model_dict = torch.load(load_path).state_dict()
            load_info = model.load_state_dict(model_dict)
            print(load_info)
        for i in range(epoch):
            print(f'---------epoch {i}---------')
            model.train()
            # result for each epoch
            for data in train_dataloader:
                to_cuda(data)
                tag_score = model(data)
                loss = Loss(tag_score, data['labels'])
                loss.backward()

                tag_pred = torch.argmax(tag_score, dim=1)
                del tag_score
                acc = accuracy_score(data['labels'].cpu().numpy(), tag_pred.cpu().numpy())

                train_step_acc.append(acc)
                train_step_loss.append(loss.item())
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
                if step % args.val_step == 0:
                    print('########### start validating ##########')
                    with torch.no_grad():
                        model.eval()
                        y_true = []
                        y_pred = []
                        for data in tqdm(val_dataloader):
                            to_cuda(data)
                            tag_score = model(data)
                            tag_pred = torch.argmax(tag_score, dim=1)
                            y_pred += tag_pred.cpu().numpy().tolist()
                            y_true += data['labels'].cpu().numpy().tolist()
                            #acc = accuracy_score(data['labels'].cpu().numpy(), tag_pred.cpu().numpy())
                            #step_val_acc.append(acc)

                        #val_acc = np.mean(step_val_acc)
                        val_acc, val_micro, val_macro = get_metrics(y_true, y_pred)
                        print('[STEP %d EVAL RESULT] accuracy: %.4f%%, micro-f:%.4f%%,\
                            macro-f:%.4f%%' % (step, val_acc*100, val_micro*100, val_macro*100))

                        if val_acc > best_acc:
                            torch.save(model, MODEL_SAVE_PATH)
                            print('Best checkpoint! checkpoint saved')
                            best_acc = val_acc

                        # save training result
                        result_data['val_acc'] = val_acc
                        result_data['val_micro'] = val_micro
                        result_data['val_macro'] = val_macro
                        result_data['train_acc'] = np.mean(step_acc)
                        result_data['train_loss'] = np.mean(step_loss)
                        resultlog.update(step, result_data)
                        print('result log saved')
                        # clear
                        step_acc = []
                        step_loss = []
                        #step_val_acc = []

                    # reset model to train mode
                    model.train()

    # test
    print('################# start testing #################')
    if args.load_ckpt is not None:
        load_path =  args.load_ckpt
    else:
        load_path = MODEL_SAVE_PATH
    model_dict = torch.load(load_path).state_dict()
    load_info = model.load_state_dict(model_dict)
    print(load_info)
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(test_dataloader):
            to_cuda(data)
            tag_score = model(data)
            tag_pred = torch.argmax(tag_score, dim=1)
            y_pred += tag_pred.cpu().numpy().tolist()
            y_true += data['labels'].cpu().numpy().tolist()
        #acc = accuracy_score(y_true, y_pred)
        acc, micro, macro = get_metrics(y_true, y_pred, isfewnerd=args.data=='fewnerd')
        resultlog.update('test_acc', {'acc':acc, 'micro':micro, 'macro':macro})
        print('[TEST RESULT] accuracy: %.4f%%, micro-f:%.4f%%, macro-f:%.4f%%' % (acc*100, micro*100, macro*100))


    

if __name__ == '__main__':
    main()


        






