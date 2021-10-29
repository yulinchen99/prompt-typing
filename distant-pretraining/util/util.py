import pandas as pd
import os
import json
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaTokenizer, BertTokenizer, GPT2Config, GPT2Tokenizer
import torch.nn as nn

def load_tag_mapping(datadir):
    filepath = os.path.join(datadir, 'tag_mapping.txt')
    df = pd.read_csv(filepath, sep='\t', header=None)
    tag_mapping = dict(zip(list(df[0]), list(df[1])))
    return tag_mapping

def get_label_ids(tokenizer, tag_mapping):
    label_ids = []
    mapped_items = sorted(tag_mapping.items(), key=lambda x:x[0])
    _, mapped_tags = zip(*mapped_items)
    mapped_splitted_tags = [list(set(tag.split('/'))) for tag in mapped_tags]
    for tag in mapped_splitted_tags:
        label_ids.append(get_tag2inputid(tokenizer, tag))
    return label_ids


def get_tag2inputid(tokenizer, splitted_tag):
    tokenized_tag = []
    for tag in splitted_tag:
        tokenized_tag.append(tokenizer.tokenize(tag)[0]) # get the first token\
    ids = tokenizer.convert_tokens_to_ids(tokenized_tag)
    return ids

def get_tokenizer(model_name):
    tokenizer = None
    #config = AutoConfig.from_pretrained(model_name,  local_files_only = True)
    #if isinstance(config, RobertaConfig):
    #    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    #elif isinstance(config, BertConfig):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    #elif isinstance(config, GPT2Config):
    #    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    #else:
    #    print('unsupported model name')
    #    raise ValueError
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