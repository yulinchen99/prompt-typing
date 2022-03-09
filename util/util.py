import pandas as pd
import os
import json
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaTokenizer, BertTokenizer, GPT2Config, GPT2Tokenizer
import torch.nn as nn
import numpy as np
import torch

def load_tag_mapping(datadir):
    filepath = os.path.join(datadir, 'tag_mapping.txt')
    print(filepath)
    df = pd.read_csv(filepath, sep='\t', header=None)
    tag_mapping = dict(zip(list(df[0]), list(df[1])))
    return tag_mapping

def get_tag_list(datadir, tag_mapping):
    filepath = os.path.join(datadir, 'tags.txt')
    mapped_tag_list = []
    ori_tag_list = []
    with open(filepath, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            ori_tag_list.append(line.strip())
            mapped_tag_list.append(tag_mapping[line.strip()])
    return ori_tag_list, mapped_tag_list

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
        self.loss = nn.BCELoss()
    
    def forward(self, score, label):
        score = sigmoid_fn(score)
        binary_label = torch.zeros(score.size()).to(score.device)
        for i, la in enumerate(label):
            binary_label[i][la] = 1
        return self.loss(score, binary_label)

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
        arg_max_ind = np.argmax(single_dist)
        pred_id = [arg_max_ind]
        pred_id.extend(
            [i for i in range(len(single_dist)) if single_dist[i] > 0.5 and i != arg_max_ind])
        pred_idx.append(pred_id)
    return pred_idx