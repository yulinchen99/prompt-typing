import sys
import argparse
from transformers import BertConfig, RobertaConfig
from util.data_loader import get_loader, EntityTypingDataset, OpenEntityDataset, OpenEntityDatasetForPrompt, OpenEntityGeneralDataset, OpenEntityGeneralDatasetForPrompt
from model.baseline import EntityTypingModel as BaselineModel
from model.maskedlm import EntityTypingModel as MaskedLM
from util.util import load_tag_mapping, get_tag2inputid, load_tag_list, ResultLog, get_tokenizer, PartialLabelLoss, MultiLabelLoss, get_output_index
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
from util.metrics import get_metrics, get_openentity_metrics, get_openentity_metrics_for_prompt
import torch.nn.functional as F
import pandas as pd
#from memory_profiler import profile

warnings.filterwarnings('ignore')

DATA_CLASS = {"default": [EntityTypingDataset, EntityTypingDataset], "openentity": [OpenEntityDataset, OpenEntityDatasetForPrompt], "openentity-general": [OpenEntityGeneralDataset, OpenEntityGeneralDatasetForPrompt]}

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
    parser.add_argument('--model_name', type=str, default='roberta-base', help='bert-base-cased, roberta-base, and gpt2 are supported, or a pretrained model save path')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default='ontonote', help='ontonote, fewnerd or bbn')
    parser.add_argument('--model', type=str, default='maskedlm', help='baseline or maskedlm')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--embed_lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=200)
    parser.add_argument('--grad_accum_step', type=int, default=1)
    parser.add_argument('--warmup_step', type=int, default=100)
    parser.add_argument('--val_step', type=int, default=2000, help='val every x steps of training')
    parser.add_argument('--log_step', type=int, default=2000, help='log every x steps of training')

    parser.add_argument('--val_iter', type=int, default=None, help='val iter')
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--result_save_dir', type=str, default='result')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--sample_num', type=int, default=None, help='default training on all samples, set a number to indicate how many samples in each type are sampled as training set')
    parser.add_argument('--calibrate', action='store_true', default=False)
    parser.add_argument('--save_result', action='store_true', default=False)




    # for soft prompt only
    parser.add_argument('--prompt', type=str, default='soft', help='soft or hard')
    #parser.add_argument('--dual_optim', action='store_true', default=False, help='set True if separate learning rate in maskedlm p-prompt setting is desired')
    parser.add_argument('--dropout', type=float, default=0.1)

    # for baseline only
    parser.add_argument('--usecls', action='store_true', default=False)
    parser.add_argument('--highlight_entity', type=str, default=None, help='for baseline model, highlight tokens around entity')
    parser.add_argument('--loss', type=str, default='cross', help='cross or partial')


    args = parser.parse_args()
    # set random seed
    set_seed(args.seed)

    # data path
    # main_dir = '/mnt/sfs_turbo/cyl/ner-mlm/'
    main_dir = sys.path[0] + '/'
    IS_FEWNERD=args.data=='fewnerd'
    if args.data == "openentity" or args.data == "openentity-general":
        args.loss = "multi_label"

    args.data = os.path.join(main_dir + 'data', args.data)
    print('data path:', args.data)

    # model saving path
    data_name = args.data.split('/')[-1]
    if '/' not in args.model_name:
        model_name = args.model_name
    else:
        model_name = '-'.join(args.model_name.split('/')[-2:])
    MODEL_SAVE_PATH = os.path.join(main_dir, args.save_dir, f'{args.model}-{model_name}-{data_name}-{args.prompt}-seed_{args.seed}-{args.sample_num}')
    if args.ckpt_name:
        MODEL_SAVE_PATH += '_' + args.ckpt_name

    # if args.dual_optim and args.model == 'maskedlm':
        # MODEL_SAVE_PATH += '-dual_optim'
    if not os.path.exists(os.path.join(main_dir, args.save_dir)):
        os.mkdir(os.path.join(main_dir, args.save_dir))
    args.model_save_path = MODEL_SAVE_PATH
    print('modelsave path:', MODEL_SAVE_PATH)
    
    # prompt
    HIGHLIGHT_ENTITY = None
    if args.highlight_entity is not None:
        HIGHLIGHT_ENTITY = args.highlight_entity.split('-')
 

    # get tag list
    print('get tag list...')
    tag_filename = "tags.txt"
    if "openentity" in args.data:
        tag_filename = "types.txt"
    ori_tag_list = load_tag_list(args.data, filename=tag_filename)
    if args.model == "baseline":
        out_dim = len(ori_tag_list)
    elif args.model == "maskedlm":
        tag_list = load_tag_list(args.data)
        tag_mapping = load_tag_mapping(args.data)
        mapped_tag_list = [tag_mapping[t] for t in tag_list]
        out_dim = len(tag_mapping)
        tag2idx = {tag:idx for idx, tag in enumerate(mapped_tag_list)}
        idx2tag = {idx:tag for idx, tag in enumerate(mapped_tag_list)}

    idx2oritag = {idx:tag for idx, tag in enumerate(ori_tag_list)}
    oritag2idx = {tag:idx for idx, tag in enumerate(ori_tag_list)}
    print(idx2oritag)
    # print(tag2idx)

    # initialize model
    print('initializing model...')
    if args.model == 'baseline':
        model = BaselineModel(args.model_name, out_dim, highlight_entity=HIGHLIGHT_ENTITY, dropout=args.dropout, usecls=args.usecls, max_length=args.max_length)
    elif args.model == 'maskedlm':
        model = MaskedLM(args.model_name, idx2tag, mapped_tag_list, prompt_mode=args.prompt, max_length=args.max_length)
    else:
        raise NotImplementedError
    model = model.cuda()

    # initialize dataloader
    print(f'initializing data from {args.data}...')
    if "openentity" in args.data:
        if args.model == "baseline":
            train_dataset = DATA_CLASS.get(data_name, DATA_CLASS["default"])[0](args.data, 'train', args.max_length, oritag2idx, highlight_entity=HIGHLIGHT_ENTITY, sample_num=args.sample_num)
            val_dataset = DATA_CLASS.get(data_name, DATA_CLASS["default"])[0](args.data, 'dev', args.max_length, oritag2idx, highlight_entity=HIGHLIGHT_ENTITY)
            test_dataset = DATA_CLASS.get(data_name, DATA_CLASS["default"])[0](args.data, 'test', args.max_length, oritag2idx, highlight_entity=HIGHLIGHT_ENTITY)
        else:
            train_dataset = DATA_CLASS.get(data_name, DATA_CLASS["default"])[1](args.data, 'train', args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY, sample_num=args.sample_num)
            val_dataset = DATA_CLASS.get(data_name, DATA_CLASS["default"])[1](args.data, 'dev', args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY)
            test_dataset = DATA_CLASS.get(data_name, DATA_CLASS["default"])[1](args.data, 'test', args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY)
    else:
        train_dataset = EntityTypingDataset(args.data, 'train', args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY, sample_num=args.sample_num)
        val_dataset = EntityTypingDataset(args.data, 'dev', args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY)
        test_dataset = EntityTypingDataset(args.data, 'test', args.max_length, tag2idx, tag_mapping, highlight_entity=HIGHLIGHT_ENTITY)

    print(train_dataset[0])

    train_dataloader = get_loader(train_dataset, args.batch_size)
    if args.sample_num is not None and len(train_dataset) < len(val_dataset):
        val_dataset, _ = torch.utils.data.random_split(val_dataset, [len(train_dataset), len(val_dataset)-len(train_dataset)], generator=torch.Generator().manual_seed(0))
    print('val dataset length ', len(val_dataset))
    val_dataloader = get_loader(val_dataset, args.val_batch_size)
    test_dataloader = get_loader(test_dataset, args.val_batch_size)

    # initialize loss
    if args.loss == 'cross':
        Loss = nn.CrossEntropyLoss()
    elif args.loss == 'partial':
        Loss = PartialLabelLoss()
    elif args.loss == "multi_label":
        Loss = MultiLabelLoss(model_type = args.model)
    else:
        assert False, print(f'invalid loss {args.loss}!')

    # initialize optimizer

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    global_train_iter = int(args.epoch * len(train_dataloader) / args.grad_accum_step + 0.5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=global_train_iter)

    # result log saving path
    result_save_dir = os.path.join(main_dir, args.result_save_dir)
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    result_save_path = os.path.join(result_save_dir, now+'.json')
    resultlog = ResultLog(args, result_save_path)

    # train
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
    best_metric = 0.0

    if not args.test_only:
        if args.load_ckpt is not None:
            print(f'loading pre-trained ckpt {args.load_ckpt}...')
            load_path =  args.load_ckpt
            model_dict = torch.load(load_path).state_dict()
            load_info = model.load_state_dict(model_dict)
            print(load_info)
        print('######### start training ##########')
        print("Model: {}".format(args.model))
        print("Total training steps: {}".format(int(len(train_dataloader)*epoch)))
        print("Learning rate: {}".format(args.lr))
        print("Validation per steps: {}".format(args.val_step))
        for i in range(epoch):
            print(f'---------epoch {i}---------')
            model.train()
            # result for each epoch
            for data in tqdm(train_dataloader, desc=f"Train Epoch {i+1}"):
                # print(data)
                to_cuda(data)
                tag_score = model(data)
                loss = Loss(tag_score, data['labels'])
                loss.backward()

                if args.loss == "multi_label":
                    tag_pred = get_output_index(tag_score)
                else: 
                    tag_pred = torch.argmax(tag_score, dim=1).cpu().numpy().tolist()

                del tag_score
                labels = data['labels']
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy().tolist()
                if args.loss == "multi_label":
                    acc, _, _ = get_openentity_metrics(labels, tag_pred, idx2oritag)
                else:
                    acc, _, _ = get_metrics(labels, tag_pred, idx2oritag, isfewnerd=IS_FEWNERD)

                train_step_acc.append(acc)
                train_step_loss.append(loss.item())
                step_acc.append(acc)
                step_loss.append(loss.item())

                step += 1

                if step % args.grad_accum_step == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if step % args.log_step == 0:
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

                        val_iter = 0
                        for data in tqdm(val_dataloader, desc="Valid"):
                            to_cuda(data)
                            tag_score = model(data)
                            if args.loss == "multi_label":
                                tag_pred = get_output_index(tag_score)
                            else: 
                                tag_pred = torch.argmax(tag_score, dim=1).cpu().numpy().tolist()
                            y_pred += tag_pred
                            labels = data['labels']
                            if isinstance(labels, torch.Tensor):
                                labels = labels.cpu().numpy().tolist()
                            y_true += labels
                            #acc = accuracy_score(data['labels'].cpu().numpy(), tag_pred.cpu().numpy())
                            #step_val_acc.append(acc)

                            val_iter += 1
                            if val_iter == args.val_iter:
                                break

                        #val_acc = np.mean(step_val_acc)
                        # val_acc, val_micro, val_macro = get_metrics(y_true, y_pred, idx2oritag, isfewnerd=IS_FEWNERD)
                        if args.loss == "multi_label":
                            if args.model == "baseline":
                                val_acc, val_micro, val_macro = get_openentity_metrics(y_true, y_pred, idx2oritag)
                            else:
                                val_acc, val_micro, val_macro = get_openentity_metrics_for_prompt(y_true, y_pred, idx2tag, idx2oritag, oritag2idx, ori_tag_list)
                        else:
                            val_acc, val_micro, val_macro = get_metrics(y_true, y_pred, idx2oritag, isfewnerd=IS_FEWNERD)
                        print('[STEP %d EVAL RESULT] accuracy: %.4f%%, micro:%s, \
                            macro:%s' % (step, val_acc*100, str(val_micro), str(val_macro)))

                        # if val_acc > best_metric:
                        #     torch.save(model, MODEL_SAVE_PATH)
                        #     print('Best checkpoint! checkpoint saved')
                        #     best_metric = val_acc

                        if val_micro["f"] > best_metric:
                            torch.save(model, MODEL_SAVE_PATH)
                            print('Best checkpoint! checkpoint saved')
                            best_metric = val_micro["f"]

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
    load_path = ''
    if args.load_ckpt is not None:
        load_path =  args.load_ckpt
    # else:
    #    load_path = MODEL_SAVE_PATH
    #    print(f'no load_ckpt designated, will load {MODEL_SAVE_PATH} automatically...')
    if load_path:
        model_dict = torch.load(load_path).state_dict()
        load_info = model.load_state_dict(model_dict)
        print(load_info)

    y_true = []
    y_pred = []
    text = []
    with torch.no_grad():
        model.eval()
        prior_dist = None

        # calibration
        if args.calibrate:
            print('get prior distribution')
            tag_dist = []
            calib_dataset = random.sample(test_dataset.samples, 1000)
            calib_dataloader = get_loader(calib_dataset, 1000)
            for data in tqdm(calib_dataloader):
                to_cuda(data)
                tag_score = model(data)
                tag_dist.append(F.softmax(tag_score))
            tag_dist = torch.cat(tag_dist, dim=0)
            prior_dist = torch.mean(tag_dist, dim=0)

        for data in tqdm(test_dataloader, desc="Test"):

            text += [' '.join(words) for words in data['words']]

            to_cuda(data)
            tag_score = model(data)

            # divide by prior_dist
            if prior_dist is not None:
                tag_score = F.softmax(tag_score) / prior_dist

            if args.loss == "multi_label":
                tag_pred = get_output_index(tag_score)
                # print(len(tag_pred[0]))
            else: 
                tag_pred = torch.argmax(tag_score, dim=1).cpu().numpy().tolist()

            y_pred += tag_pred
            labels = data['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy().tolist()
            y_true += labels
        #acc = accuracy_score(y_true, y_pred)
        if 'ontonote' in args.data:
            y_true = torch.LongTensor(y_true)
            y_pred = torch.LongTensor(y_pred)
            y_pred = y_pred[y_true != tag2idx['other']]
            y_true = y_true[y_true != tag2idx['other']]
            y_true = y_true.numpy().tolist()
            y_pred = y_pred.numpy().tolist()
        # print(y_true[:100])
        # print(y_pred[:100])
        if args.loss == "multi_label":
            if args.model == "baseline":
                acc, micro, macro = get_openentity_metrics(y_true, y_pred, idx2oritag)
            else: # TODO 
                print("calculating metrics...")
                acc, micro, macro = get_openentity_metrics_for_prompt(y_true, y_pred, idx2tag, idx2oritag, oritag2idx, ori_tag_list)

        else:
            acc, micro, macro = get_metrics(y_true, y_pred, idx2oritag, isfewnerd=IS_FEWNERD)
        # acc, micro, macro = get_metrics(y_true, y_pred, idx2oritag, isfewnerd=IS_FEWNERD)
        resultlog.update('test_acc', {'acc':acc, 'micro':micro, 'macro':macro})
        print('[TEST RESULT] accuracy: %.4f%%, micro:%s, macro:%s' % (acc*100, str(micro), str(macro)))

        # if args.save_result:
        #     d = {'text':text, 'label':[idx2oritag[idx] for idx in y_true], 'pred':[idx2oritag[idx] for idx in y_pred]}
        #     d = pd.DataFrame(d)
        #     d.to_csv(os.path.join(main_dir, model_name+load_path+'-testresult.csv'))
        #     print('test result saved')
                
    

if __name__ == '__main__':
    main()


        






