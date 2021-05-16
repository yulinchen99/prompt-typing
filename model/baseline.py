import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaConfig, BertConfig, RobertaModel, BertModel, RobertaTokenizer, BertTokenizer
import sys
sys.path.append('../')
from util.util import get_tag2inputid

class EntityTypingModel(nn.Module):
    def __init__(self, model_name, idx2tag, tag_list, out_dim=None, highlight_entity=None):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif isinstance(config, BertConfig):
            self.model = BertModel(config).from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            print('unsupported model name')
        if highlight_entity:
            added_num = self.tokenizer.add_tokens(highlight_entity)
            self.model.resize_token_embeddings(config.vocab_size+added_num)
        self.model = nn.DataParallel(self.model)
        self.idx2tag = idx2tag
        self.tag2inputid = get_tag2inputid(self.tokenizer, tag_list)
        self.linear = nn.Linear(768, out_dim)

    def __get_tag_score__(self, output, inputs):
        out_score = self.linear(output.last_hidden_state)
        # get score at position [ENTITY]
        tag_score = []
        for i, score in enumerate(out_score):
            tag_score.append(score[inputs['entity_pos'][i][0]-1])
        return torch.stack(tag_score)

    def forward(self, inputs):
        # tokenize
        output = self.tokenizer(inputs['words'], is_split_into_words=True, return_attention_mask=True, return_tensors='pt', padding=True)
        tag_output = self.model(input_ids=output['input_ids'].cuda(), attention_mask=output['attention_mask'].cuda(), output_hidden_states=True)
        tag_score = self.__get_tag_score__(tag_output, inputs)
        return tag_score