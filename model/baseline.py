import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaModel, BertModel, RobertaTokenizer, BertTokenizer, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import sys
sys.path.append('../')
from util.util import get_tag2inputid

class EntityTypingModel(nn.Module):
    def __init__(self, model_name, idx2tag, tag_list, out_dim=None, highlight_entity=None, dropout=0.1, usecls=False):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout
        if isinstance(config, RobertaConfig):
            self.model = RobertaModel(config).from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif isinstance(config, BertConfig):
            self.model = BertModel(config).from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif isinstance(config, GPT2Config):
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        else:
            print('unsupported model name')
            raise ValueError
        if highlight_entity:
            added_num = self.tokenizer.add_tokens(highlight_entity)
            self.model.resize_token_embeddings(config.vocab_size+added_num)
        self.model = nn.DataParallel(self.model)
        self.idx2tag = idx2tag
        self.tag2inputid = get_tag2inputid(self.tokenizer, tag_list)
        self.linear = nn.Linear(config.hidden_size, out_dim)
        self.usecls = usecls

    def __get_tag_score__(self, output, inputs):
        # get score at position [ENTITY]
        if not self.usecls:
            tag_score = []
            out_score = self.linear(output.last_hidden_state)
            for i, score in enumerate(out_score):
                tag_score.append(score[inputs['entity_pos'][i][0]-1])
            tag_score = torch.stack(tag_score)
            tag_score = self.linear(tag_score)
        else:
            tag_score = self.linear(output.last_hidden_state[:,0,:])
        return tag_score

    def forward(self, inputs, use_sep=True):
        # tokenize
        if use_sep:
            words = inputs['words']
            for i in range(len(words)):
                pos = inputs['entity_pos'][i]
                words[i] = words[i] + [self.tokenizer.sep_token] + words[i][pos[0]:pos[1]]
            output = self.tokenizer(words, is_split_into_words=True, return_attention_mask=True, return_tensors='pt', padding=True)
        else:
            output = self.tokenizer(inputs['words'], is_split_into_words=True, return_attention_mask=True, return_tensors='pt', padding=True)
        tag_output = self.model(input_ids=output['input_ids'].cuda(), attention_mask=output['attention_mask'].cuda(), output_hidden_states=True)
        tag_score = self.__get_tag_score__(tag_output, inputs)
        return tag_score