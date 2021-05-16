# %%
import torch
import torch.utils.data as data
import os
import numpy as np
from transformers import BertTokenizer, AutoConfig, RobertaConfig, BertConfig, RobertaTokenizer
# from word_encoder import BERTWordEncoder

class Sample:
    def __init__(self, words, tag, label, pos):
        self.words = words
        self.tag = tag
        self.label = label
        self.pos = list(pos)
    
    def highlight(self, highlight_tokens):
        self.words.insert(self.pos[0], highlight_tokens[0])
        self.words.insert(self.pos[1]+1, highlight_tokens[1])
        # update pos
        self.pos = [self.pos[0]+1, self.pos[1]+1]

    def valid(self, max_length):
        return self.pos[1] <= max_length and len(self.words) <= max_length


class EntityTypingDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, filepath, max_length, tag2idx, tag_mapping, highlight_entity=None):
        if not os.path.exists(filepath):
            print(f"[ERROR] Data file {filepath} does not exist!")
            assert(0)
        self.samples = []
        self.max_length = max_length
        self.tag2id = tag2idx
        self.tag_mapping = tag_mapping
        self.highlight_entity = highlight_entity
        self.__load_data_from_file__(filepath)
    
    def __load_data_from_file__(self, filepath):
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        for line in lines:
            linelist = line.strip().split('\t')
            start = int(linelist[0])
            end = int(linelist[1])
            tag = linelist[3]
            # map tags
            tag = self.tag_mapping[tag.lower()]
            label = self.tag2id[tag]
            words = linelist[2].split(' ')
            sample = Sample(words, tag, label, (start, end))
            if sample.valid(self.max_length):
                self.samples.append(sample)
            
            # add <ENTITY> </ENTITY> 
        if self.highlight_entity:
            for sample in self.samples:
                sample.highlight(self.highlight_entity)
    
    def __getitem__(self, index):
        # get raw data
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def collate_fn(samples):
    batch_data = {'words':[], 'labels':[], 'entity_pos':[]}
    for sample in samples:
        batch_data['words'].append(sample.words)
        batch_data['labels'].append(sample.label)
        batch_data['entity_pos'].append(sample.pos)
    batch_data['labels'] = torch.LongTensor(batch_data['labels'])
    batch_data['entity_pos'] = torch.LongTensor(batch_data['entity_pos'])
    return batch_data

def get_loader(datadir, mode, batch_size, max_length, tag2idx, tag_mapping, num_workers=8, collate_fn=collate_fn, highlight_entity=None):
    filepath = os.path.join(datadir, f'{mode}.txt')
    dataset = EntityTypingDataset(filepath, max_length, tag2idx, tag_mapping, highlight_entity=highlight_entity)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader
# %%
