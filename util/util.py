import pandas as pd
import os
import json
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaTokenizer, BertTokenizer, GPT2Config, GPT2Tokenizer
import torch.nn as nn
import numpy as np
import torch

TYPE_NUM_DICT = {"open": 10331, "onto": 89, "wiki": 4600, "baseline": {"kb": 130, "gen": 9}, "maskedlm": {"kb": 138, "gen": 9}}

def load_tag_mapping(datadir):
    filepath = os.path.join(datadir, 'tag_mapping.txt')
    print(filepath)
    df = pd.read_csv(filepath, sep='\t', header=None)
    tag_mapping = dict(zip(list(df[0]), list(df[1])))
    return tag_mapping

def load_tag_list(datadir, filename="tags.txt"):
    filepath = os.path.join(datadir, filename)
    # mapped_tag_list = []
    ori_tag_list = []
    with open(filepath, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            ori_tag_list.append(line.strip())
            # mapped_tag_list.append(tag_mapping[line.strip()])
    return ori_tag_list

def get_tag2inputid(tokenizer, mapped_tag_list):
    splitted_tags = [list(set(tag.split('/'))) for tag in mapped_tag_list]

    # tokenize
    tokenized_tags = []
    for splitted_tag in splitted_tags:
        tokenized_tag = []
        for tag in splitted_tag:
            tokenized_tag.append(tokenizer.tokenize(tag)[0]) # get the first token
        tokenized_tags.append(tokenized_tag) 

    d ={}
    for i, tokenized_tag in enumerate(tokenized_tags):
        d[mapped_tag_list[i]] = tokenizer.convert_tokens_to_ids(tokenized_tag)
    return d

def get_tokenizer(model_name):
    tokenizer = None
    config = AutoConfig.from_pretrained(model_name)
    if isinstance(config, RobertaConfig):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif isinstance(config, BertConfig):
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif isinstance(config, GPT2Config):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        print('unsupported model name')
        raise ValueError
    return tokenizer

class ResultLog:
    '''
    create json log
    {
        model_name:
        train_file:
        val_file:
        test_file:
        seed:
        model_save_dir:
        result: {
            0: {
                train_loss: 
                train_acc: 
                val_acc: 
            }
            ...
        }
    }
    '''
    def __init__(self, args, save_path):
        d = vars(args)
        d['result'] = {}
        with open(save_path, 'w')as f:
            f.write(json.dumps(d))
        self.save_path = save_path

    def update(self, epoch, data):
        with open(self.save_path, 'r+')as f:
            d = json.load(f)
            f.seek(0)
            f.truncate()
            d['result'][epoch] = data
            f.write(json.dumps(d))

    def delete(self):
        os.remove(self.save_path)

class PartialLabelLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.logsoftmax = nn.LogSoftmax()
    
    def forward(self, score, label):
        assert score.size(0) == label.size(0)
        score = score.float()
        score = self.logsoftmax(score)
        loss = 0.0
        for i in range(label.size(0)):
            loss -= score[i][label[i]]
        loss = loss / label.size(0)
        return loss

sigmoid_fn = nn.Sigmoid()

class MultiLabelLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, score, label, model_type="baseline"):
        binary_label = torch.zeros(score.size()).to(score.device)
        for i, la in enumerate(label):
            binary_label[i][la] = 1.0
        if model_type == "maskedlm": # use simple BCE loss for maskedlm
            return self.loss(score, binary_label)
        else:
            return self.multilabel_bin_loss(score, binary_label, model_type)
    
    # def onehot_encode_batch(self, class_ids_list, n_classes):
    #     batch_size = len(class_ids_list)
    #     tmp = np.zeros((batch_size, n_classes), dtype=np.float32)
    #     for i, class_ids in enumerate(class_ids_list):
    #         for cid in class_ids:
    #             tmp[i][cid] = 1.0
    #     return tmp

    def multilabel_bin_loss(self, logits, targets, model_type):
        loss_func = self.loss
        d = TYPE_NUM_DICT[model_type]
        gen_cutoff, fine_cutoff, final_cutoff = d['gen'], d['kb'], None
        loss_is_valid = False
        loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
        comparison_tensor = torch.tensor([1.0], device=logits.device)
        gen_targets = targets[:, :gen_cutoff]
        fine_targets = targets[:, gen_cutoff:fine_cutoff]
        gen_target_sum = torch.sum(gen_targets, 1)
        fine_target_sum = torch.sum(fine_targets, 1)
        # print(logits.size())
        # print(targets)
        # print(gen_target_sum)
        # print(gen_target_sum.size())

        if torch.sum(gen_target_sum.data) > 0:
            gen_mask = torch.squeeze(torch.nonzero(
                torch.min(gen_target_sum.data, comparison_tensor), as_tuple=False), dim=1)
            # print(gen_mask)
            gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
            # gen_mask = torch.autograd.Variable(gen_mask).cuda()
            gen_target_masked = gen_targets.index_select(0, gen_mask)
            # print(gen_target_masked)
            # exit()
            gen_loss = loss_func(gen_logit_masked, gen_target_masked)
            loss += gen_loss
            loss_is_valid = True
        if torch.sum(fine_target_sum.data) > 0:
            fine_mask = torch.squeeze(torch.nonzero(
                torch.min(fine_target_sum.data, comparison_tensor), as_tuple=False), dim=1)
            fine_logit_masked = logits[:, gen_cutoff:fine_cutoff][fine_mask, :]
            # fine_mask = torch.autograd.Variable(fine_mask).cuda()
            fine_target_masked = fine_targets.index_select(0, fine_mask)
            fine_loss = loss_func(fine_logit_masked, fine_target_masked)
            loss += fine_loss
            loss_is_valid = True

        if final_cutoff:
            finer_targets = targets[:, fine_cutoff:final_cutoff]
            logit_masked = logits[:, fine_cutoff:final_cutoff]
        else:
            logit_masked = logits[:, fine_cutoff:]
            finer_targets = targets[:, fine_cutoff:]
        if torch.sum(torch.sum(finer_targets, 1).data) > 0:
            finer_mask = torch.squeeze(torch.nonzero(
                torch.min(torch.sum(finer_targets, 1).data, comparison_tensor), as_tuple=False), dim=1)
            # finer_mask = torch.autograd.Variable(finer_mask).cuda()
            finer_target_masked = finer_targets.index_select(0, finer_mask)
            logit_masked = logit_masked[finer_mask, :]
            layer_loss = loss_func(logit_masked, finer_target_masked)
            loss += layer_loss
            loss_is_valid = True
        return loss if loss_is_valid else None

def get_output_index(outputs):
    """
    Given outputs from the decoder, generate prediction index.
    :param outputs:
    :return:
    """
    pred_idx = []
    outputs = sigmoid_fn(outputs).data.cpu().clone()
    for single_dist in outputs:
        single_dist = single_dist.numpy()
        pred_id = np.where(single_dist > 0.5)[0].tolist()
        if not pred_id:
            arg_max_ind = np.argmax(single_dist)
            pred_id = [arg_max_ind]
        # pred_id.extend(
            # [i for i in range(len(single_dist)) if single_dist[i] > 0.5 and i != arg_max_ind])
        pred_idx.append(pred_id)
    return pred_idx

