import torch.nn as nn
import random
import torch
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaForMaskedLM, BertForMaskedLM, RobertaTokenizer, BertTokenizer, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch.nn.functional as F

random.seed(0)

def normalize(data):
    return torch.nn.functional.normalize(data, dim=1)

class PretrainModel(nn.Module):
    def __init__(self, model_name, alpha=0.7, blank_token='[BLANK]', device='cuda:0', label_ids=None):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            #self.word_embedding = self.model.roberta.get_input_embeddings()
        elif isinstance(config, BertConfig):
            self.model = BertForMaskedLM.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            #self.word_embedding = self.model.bert.get_input_embeddings()
        elif isinstance(config, GPT2Config):
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        else:
            print('unsupported model name')
            raise ValueError
        self.alpha = alpha
        self.blank_token = blank_token
        self.device = device

        num_added_tokens = self.tokenizer.add_tokens([self.blank_token])
        self.model.resize_token_embeddings(config.vocab_size + num_added_tokens)
        if 'cuda' in device:
            self.model = nn.DataParallel(self.model)

        self.label_ids = label_ids

    def get_prompt_sentence(self, sent, pos, entity_name):
        # replace entities
        prob = random.random()
        if prob < self.alpha:
            new_sent = sent[:pos[0]] + [self.blank_token] + sent[pos[-1]+1:]
            new_sent += [self.blank_token]
        else:
            new_sent = sent + entity_name.split(' ')

        # prompt
        new_sent += ['is']
        if not isinstance(self.tokenizer, GPT2Tokenizer):
            new_sent = new_sent + [self.tokenizer.mask_token]
        
        return new_sent

    
    def tokenize(self, sent_list, pos, entity_name):
        new_sent_list = []
        for i , sent in enumerate(sent_list):
            new_sent = self.get_prompt_sentence(sent, pos[i], entity_name[i])
            new_sent_list.append(new_sent)
        # tokenize
        if isinstance(self.tokenizer, GPT2Tokenizer):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_sent = self.tokenizer(new_sent_list, is_split_into_words=True, return_tensors='pt', padding=True)
        return tokenized_sent['input_ids'].to(self.device), tokenized_sent['attention_mask'].to(self.device)

    def get_mask_logits(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pred_pos = -2 # the token position for token prediction

        if isinstance(self.tokenizer, GPT2Tokenizer):
            pred_pos = -1

        mask_logits_all = []
        logits = output.logits
        #print(logits.shape)
        for i, logit in enumerate(logits):
            cur_mask_logits_all = logit[attention_mask[i]==1,:][pred_pos].unsqueeze(0)
            mask_logits_all.append(cur_mask_logits_all)
        mask_logits_all = torch.cat(mask_logits_all, dim=0) # (batch_size, vocab_size)
        #print(mask_logits_all.shape)

        mask_logits = []
        for label_id in self.label_ids:
            mask_logits.append(torch.mean(mask_logits_all[:,label_id], dim=1).unsqueeze(-1)) # (batch_size, 1)
        mask_logits = torch.cat(mask_logits, dim=-1) # (batch_size, label_size)
        #print(mask_logits.shape)
        return mask_logits

    def get_mask_hidden_state(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pred_pos = -2 # the token position for token prediction

        if isinstance(self.tokenizer, GPT2Tokenizer):
            pred_pos = -1
        mask_hidden_states = []
        hidden_states = output.hidden_states[1]
        for i, state in enumerate(hidden_states):
            mask_hidden_states.append(state[attention_mask[i]==1,:][pred_pos].unsqueeze(0))
        mask_hidden_states = torch.cat(mask_hidden_states, dim=0)
        return mask_hidden_states

    def save(self, save_path):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)

    def load(self, load_path):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.from_pretrained(load_path)
        else:
            self.model.from_pretrained(load_path)


    
    def forward(self, inputs):
        # inputs:{'sent1':List[List[str]], 'sent2':List[List[str]], 
        #'pos1': List[List[int]], 'pos2': List[List[int]], 'entity_name':str}
        sent1_input, sent1_mask = self.tokenize(inputs['sent1'], inputs['pos1'], inputs['entity_name'])
        sent2_input, sent2_mask = self.tokenize(inputs['sent2'], inputs['pos2'], inputs['entity_name'])
        if not self.label_ids:
            sent1_hidden_state = self.get_mask_hidden_state(sent1_input, sent1_mask)
            sent2_hidden_state = self.get_mask_hidden_state(sent2_input, sent2_mask)
        else:
            sent1_hidden_state = self.get_mask_logits(sent1_input, sent1_mask)
            sent2_hidden_state = self.get_mask_logits(sent2_input, sent2_mask)
            #print(sent1_hidden_state.shape)
        # compute score
        score = F.cosine_similarity(sent1_hidden_state, sent2_hidden_state)
        #print(score)
        #p = 1.0 - 1.0 / (1.0 + torch.exp(score))
        p = 0.5 + 0.5 * score
        return sent1_hidden_state, sent2_hidden_state, p


class MTBLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, sent1_embed, sent2_embed, p, labels):
        loss = torch.sum(torch.mul(labels, torch.log(p+1e-6)) + torch.mul((1-labels), torch.log(1-p+1e-6)))
        return - loss / sent1_embed.size(0)
