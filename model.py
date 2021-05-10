# %%
import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaForMaskedLM, BertForMaskedLM
class MaskedModel(nn.Module):
    def __init__(self, model_name, out_dim):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
        elif isinstance(config, BertConfig):
            self.model = BertForMaskedLM.from_pretrained(model_name)
        else:
            print('unsupported model name')
        self.linear = nn.Linear(768, out_dim)
        self.model = nn.DataParallel(self.model)

    def __get_tag_score__(self, out_score, inputs):
        tag_score = []
        mask = inputs['tag_mask']
        for idx, score in enumerate(out_score):
            tag_score.append(score[mask[idx]==1][-2])
        tag_score = torch.stack(tag_score, dim=0)
        return tag_score

    def forward(self, inputs):
        word_loss = self.model(input_ids=inputs['word_input'], attention_mask=inputs['word_mask'], labels=inputs['word_labels']).loss
        tag_output = self.model(input_ids=inputs['tag_input'], attention_mask=inputs['tag_mask'],output_hidden_states=True)
        out_score = tag_output.hidden_states[-1]
        out_score = self.linear(out_score)
        tag_score = self.__get_tag_score__(out_score, inputs)
        return torch.mean(word_loss), tag_score