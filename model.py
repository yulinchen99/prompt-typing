# %%
import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaForMaskedLM, BertForMaskedLM
class MaskedModel(nn.Module):
    def __init__(self, model_name, idx2tag, tag2inputid, out_dim=None, uselinear=False):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
        elif isinstance(config, BertConfig):
            self.model = BertForMaskedLM.from_pretrained(model_name)
        else:
            print('unsupported model name')
        self.model = nn.DataParallel(self.model)
        self.idx2tag = idx2tag
        self.tag2inputid = tag2inputid
        self.uselinear= uselinear
        if self.uselinear:
            self.linear = nn.Linear(768, out_dim)

    def __get_tag_logits__(self, out_logits):
        tag_logits = []
        for i in range(len(self.idx2tag)):
            tag_logits.append(torch.mean(out_logits[:,self.tag2inputid[self.idx2tag[i]]], dim=1, keepdim=False).unsqueeze(-1)) #(batch_num, 1)
        return torch.cat(tag_logits, dim=-1) # (batch_num, tag_num)

    def __get_tag_score__(self, output, inputs):
        if self.uselinear:
            out_score = self.linear(output.hidden_states[-1])
            # get score at position [MASK]
            tag_score = []
            mask = inputs['tag_mask']
            for idx, score in enumerate(out_score):
                tag_score.append(score[mask[idx]==1][-2].unsqueeze(0))
            tag_score = torch.cat(tag_score, dim=0)
        else:
            out_score = output.logits
            # get score at position [MASK]
            tag_score = []
            mask = inputs['tag_mask']
            for idx, score in enumerate(out_score):
                tag_score.append(score[mask[idx]==1][-2].unsqueeze(0))
            tag_score = torch.cat(tag_score, dim=0) # (num_batch, vocab_size)
            # get score for each tag in tag space
            tag_score = self.__get_tag_logits__(tag_score)
        return tag_score

    def forward(self, inputs, tag_only=True):
        tag_output = self.model(input_ids=inputs['tag_input'], attention_mask=inputs['tag_mask'],output_hidden_states=True)
        tag_score = self.__get_tag_score__(tag_output, inputs)
        if not tag_only:
            word_loss = self.model(input_ids=inputs['word_input'], attention_mask=inputs['word_mask'], labels=inputs['word_labels']).loss
            return torch.mean(word_loss), tag_score
        else:
            return tag_score