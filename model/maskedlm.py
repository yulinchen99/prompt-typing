import sys
sys.path.append('../')
from util.util import get_tag2inputid
import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaForMaskedLM, BertForMaskedLM, RobertaTokenizer, BertTokenizer
import torch.nn.functional as F

class EntityTypingModel(nn.Module):
    def __init__(self, model_name, idx2tag, tag_list, prompt=None, is_p_prompt=True, dropout=0.1):
        # prompt: a list of words

        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.word_embedding = self.model.roberta.get_input_embeddings()
        elif isinstance(config, BertConfig):
            self.model = BertForMaskedLM.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.word_embedding = self.model.bert.get_input_embeddings()
        else:
            print('unsupported model name')
        self.prompt = prompt

        # whether [P]-[P1]-[P2] prompt or 'is' prompt
        self.is_p_prompt = is_p_prompt
        if self.is_p_prompt:
            self.prompt_embedding = nn.Embedding(len(prompt), config.hidden_size)
            self.prompt_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.model.resize_token_embeddings(config.vocab_size + len(prompt))

        self.model = nn.DataParallel(self.model)
        self.idx2tag = idx2tag
        self.tag2inputid = get_tag2inputid(self.tokenizer, tag_list)

    def __get_tag_logits__(self, out_logits):
        tag_logits = []
        for i in range(len(self.idx2tag)):
            tag_logits.append(torch.mean(out_logits[:,self.tag2inputid[self.idx2tag[i]]], dim=1, keepdim=False).unsqueeze(-1)) #(batch_num, 1)
        return torch.cat(tag_logits, dim=-1) # (batch_num, tag_num)

    def __get_tag_score__(self, output, mask):
        out_score = output.logits
        # get score at position [MASK]
        tag_score = []
        for idx, score in enumerate(out_score):
            tag_score.append(score[mask[idx]==1][-2].unsqueeze(0))
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
        if self.is_p_prompt:
            # embed prompt
            prompt_ids = torch.LongTensor([idx for idx, p in enumerate(self.prompt)]).cuda()
            prompt_embedding = self.dropout(self.prompt_linear(self.prompt_embedding(prompt_ids)))
            # embed word
            output = self.tokenizer(inputs['words'], is_split_into_words=True, return_attention_mask=True, return_tensors='pt', padding=True)
            word_embedding = self.word_embedding(output['input_ids'].cuda())
            # embed mask
            mask_embedding = self.word_embedding((torch.LongTensor([self.tokenizer.mask_token_id]).cuda()))
            # concat
            all_embedding, all_mask = self.concat_word_prompt_embedding(word_embedding, prompt_embedding, mask_embedding, output['attention_mask'].cuda(), inputs)
            tag_output = self.model(inputs_embeds=all_embedding, attention_mask=all_mask, output_hidden_states=True)
        else:
            input_words = []
            for i, words in enumerate(inputs['words']):
                pos = inputs['entity_pos'][i]
                input_words.append(words + words[pos[0]:pos[1]] + self.prompt + [self.tokenizer.mask_token])
            inputs = self.tokenizer(input_words, is_split_into_words=True, return_tensors='pt', padding=True)
            all_mask = inputs['attention_mask'].cuda()
            tag_output = self.model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), output_hidden_states=True)
        
        tag_score = self.__get_tag_score__(tag_output, all_mask)
        tag_score = self.__get_tag_logits__(tag_score)
        return tag_score
# %%
