import sys
sys.path.append('../')
from util.util import get_tag2inputid
import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaForMaskedLM, BertForMaskedLM, RobertaTokenizer, BertTokenizer, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import os
class Prompt:
    def __init__(self):
        # each prompt is [separation_token, prompt_word1, prompt_word2, ...]
        self.prompt_dict = {
            'soft':['[P1]','[P2]'],
            'hard3':['is', 'a'], 
            'hard1':['is'],
            'hard2':['is', 'a'],
            'soft1': ['[P1]', '[P2]', '[P3]'],
            'soft2': ['[P1]', '[P2]', '[P3]', '[P4]'],
            'soft3': ['[P1]', '[P2]', '[P3]', '[P4]', '[P5]']

            
        }
        self.sep_dict = {
            'soft': ['[P]'],
            'hard3': ['In', 'this', 'sentence', ','],
            'hard2':[''],
            'hard1':[''],
            'soft1': ['[P]'],
            'soft2': ['[P]'],
            'soft3': ['[P]'],
        }
        self.prompt = None

    def get_tokens(self, prompt_mode):
        if prompt_mode not in self.prompt_dict:
            print(f'no prompt for mode {prompt_mode}')
            raise ValueError
        else:
            tokens = self.sep_dict[prompt_mode] + self.prompt_dict[prompt_mode]
            tokens = [t for t in tokens if t]
            return tokens
    
    def get_prompt_sentence(self, words, pos, prompt_mode):
        prompt = self.prompt_dict[prompt_mode]
        sep = self.sep_dict[prompt_mode]
        newwords = words + sep + words[pos[0]:pos[1]] + prompt
        newwords = [w for w in newwords if w]
        return newwords

class EntityTypingModel(nn.Module):
    def __init__(self, model_name, idx2tag, tag_list, prompt_mode, max_length=128):
        nn.Module.__init__(self)
        # config = AutoConfig.from_pretrained(os.path.join(model_name, 'config.json'))
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            #self.word_embedding = self.model.roberta.get_input_embeddings()
        elif isinstance(config, BertConfig):
            self.model = BertForMaskedLM.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            #self.word_embedding = self.model.bert.get_input_embeddings()
        elif isinstance(config, GPT2Config):
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            print('unsupported model name')
            raise ValueError
        
        # handle new tokens in prompt
        self.prompt = Prompt()
        self.prompt_mode = prompt_mode
        new_tokens = self.prompt.get_tokens(prompt_mode)
        num_added_tokens = self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(config.vocab_size + num_added_tokens)

        self.model = nn.DataParallel(self.model)
        self.idx2tag = idx2tag
        self.tag2inputid = get_tag2inputid(self.tokenizer, tag_list)
        self.max_length = max_length

    def __get_tag_logits__(self, out_logits):
        tag_logits = []
        for i in range(len(self.idx2tag)):
            tag_logits.append(torch.mean(out_logits[:,self.tag2inputid[self.idx2tag[i]]], dim=1, keepdim=False).unsqueeze(-1)) #(batch_num, 1)
        return torch.cat(tag_logits, dim=-1) # (batch_num, tag_num)

    def __get_tag_score__(self, output, mask):
        pred_pos = -2 # the token position for token prediction
        if isinstance(self.tokenizer, GPT2Tokenizer):
            pred_pos = -1
        
        out_score = output.logits
        tag_score = []
        for idx, score in enumerate(out_score):
            tag_score.append(score[mask[idx]==1][pred_pos].unsqueeze(0))
        tag_score = torch.cat(tag_score, dim=0)
        return tag_score

    def concat_word_prompt_embedding(self, word_embed, prompt_embed, mask_embed, word_attention_mask, inputs):
        pos = inputs['entity_pos']
        all_embedding = []
        length = []
        for idx, sent_embed in enumerate(word_embed): # (batch_size, seq_len, embed_num)
            # strip
            sent_embed = sent_embed[word_attention_mask[idx]==1]
            # separate [SEP]
            cls_embed = sent_embed[0].unsqueeze(0)
            sep_embed = sent_embed[-1].unsqueeze(0)
            sent_embed = sent_embed[1:-1]
            # pad & shorten
            cur_len = len(cls_embed) + len(sent_embed) + len(prompt_embed) + pos[idx][1] - pos[idx][0]
            if cur_len > self.max_length:
                sent_embed = sent_embed[:-(cur_len - self.max_length)]
            
            # concat embeddings
            new_embedding = torch.cat([cls_embed, sent_embed, prompt_embed[0].unsqueeze(0), sent_embed[pos[idx][0]:pos[idx][1]], prompt_embed[1:], mask_embed, sep_embed]) # (new-seq_len, embed_num)
            length.append(new_embedding.size(0))
            all_embedding.append(new_embedding)

        # padding to longest
        max_length = max(length)
        def pad(seq):
            pad_num = (0, 0, 0, max_length-seq.size(0))
            seq = F.pad(seq, pad_num)
            return seq

        for i in range(len(all_embedding)):
            all_embedding[i] = pad(all_embedding[i])
        # stack
        all_embedding = torch.stack(all_embedding)
        # mask
        all_mask = (all_embedding.sum(dim=-1) != 0).long()
        return all_embedding, all_mask


    def forward(self, inputs):
        input_words = []
        for i, words in enumerate(inputs['words']):
            pos = inputs['entity_pos'][i]
            newwords = self.prompt.get_prompt_sentence(words, pos, self.prompt_mode)
            if not isinstance(self.tokenizer, GPT2Tokenizer):
                newwords += [self.tokenizer.mask_token]
            input_words.append(newwords)
        if isinstance(self.tokenizer, GPT2Tokenizer):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(input_words, is_split_into_words=True, return_tensors='pt', padding=True)
        all_mask = inputs['attention_mask'].cuda()
        #try:
        tag_output = self.model(input_ids=inputs['input_ids'].cuda(), attention_mask=all_mask, output_hidden_states=True)
        #except:
            #print(inputs['input_ids'].shape)
            #print(inputs['attention_mask'].shape)

        
        tag_score = self.__get_tag_score__(tag_output, all_mask)
        tag_score = self.__get_tag_logits__(tag_score)
        return tag_score
# %%
