# %%
import torch
import torch.utils.data as data
import os
import numpy as np
# from word_encoder import BERTWordEncoder

import random
import json

from util.fewshotsampler import FewshotSampler
from tqdm import tqdm

def sample_by_ratio(datalist, ratio):
    sample_num = max(1, int(len(datalist)*ratio+0.5)) # at least one
    return list(random.sample(datalist, sample_num))

def sample_by_number(datalist, number):
    sample_num = min(len(datalist), number)
    return list(random.sample(datalist, sample_num))

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
    
    def __str__(self):
        return f"Example\nwords: {self.words}\ntag: {self.tag}\nlabel: {self.label}\npos: {self.pos}\n"


class EntityTypingDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, datadir, mode, max_length, tag2idx, tag_mapping=None, highlight_entity=None, sample_num=None):
        filepath = self._get_filepath(datadir, mode)
        self.samples = []
        self.max_length = max_length
        self.tag2id = tag2idx
        self.tag_mapping = tag_mapping
        self.highlight_entity = highlight_entity
        self.sample_num = sample_num
        self.__load_data_from_file__(filepath)
        self.__sample_data__()

    def _get_filepath(self, datadir, mode):
        return os.path.join(datadir, f'{mode}.txt')

    def __sample_data__(self):
        if self.sample_num is None:
            return
        # for each type of entity, sample by sample_num
        sample_group = {}
        for sample in self.samples:
            if sample.tag in sample_group:
                sample_group[sample.tag].append(sample)
            else:
                sample_group[sample.tag] = [sample]
        for tag in sample_group:
            sample_group[tag] = sample_by_number(sample_group[tag], self.sample_num)
        self.samples = [sample for sample_list in list(sample_group.values()) for sample in sample_list]
    
    def __load_data_from_file__(self, filepath):
        if not os.path.exists(filepath):
            print(f"[ERROR] Data file {filepath} does not exist!")
            assert(0)
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        for line in lines:
            linelist = line.strip().split('\t')
            start = int(linelist[0])
            end = int(linelist[1])
            tag = linelist[3]
            # map tags
            if self.tag_mapping:
                tag = self.tag_mapping[tag]
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

gen_tags = ["person", "group", "organization", "location", "entity", "time", "object", "event", "place"]
class OESample(Sample):
    def get_class_count(self):
        if hasattr(self, "_gen_tags") and self._gen_tags is not None:
            return self._gen_tags
        tags = list(set(gen_tags).intersection(set(self.tag)))
        self._gen_tags = dict(zip(tags, [1]*len(tags)))
        return self._gen_tags

class OpenEntityDataset(EntityTypingDataset):
    def _get_tags(self, d):
        return d["y_str"]

    def __load_data_from_file__(self, filepath):
        for file in filepath:
            if not os.path.exists(file):
                print(f"[ERROR] Data file {filepath} does not exist!")
                assert(0)
            with open(file, 'r', encoding='utf-8')as f:
                lines = f.readlines()
            for line in tqdm(lines, desc="load"):
                d = json.loads(line)
                tag = self._get_tags(d)
                start = len(d["left_context_token"])
                mention = d["mention_span"].split(" ")
                end = start + len(mention)
                label = [self.tag2id[t] for t in tag if t in self.tag2id]
                words = d["left_context_token"] + mention + d["right_context_token"]
                sample = OESample(words[:self.max_length], tag, label, (start, end))
                if sample.valid(self.max_length):
                    self.samples.append(sample)
            
        # add <ENTITY> </ENTITY> 
        if self.highlight_entity:
            for sample in self.samples:
                sample.highlight(self.highlight_entity)
        
        print(f"{len(self.samples)} samples loaded.")
        
    def _get_filepath(self, datadir, mode):
        if mode == "test" or mode == "dev" or mode == "train":
            return [os.path.join(datadir, f'{mode}.json')]
        else:
            return [os.path.join(datadir, f'{mode}.json'), os.path.join(datadir, f"el_{mode}.json"), os.path.join(datadir, f"headword_{mode}.json")]

    
    def __sample_data__(self):
        if self.sample_num is None:
            return
        # for each type of entity, sample by sample_num
        sampler = FewshotSampler(self.sample_num, samples=self.samples)
        sampled_idx = sampler.__next__()
        self.samples = [self.samples[i] for i in sampled_idx]

class OpenEntityDatasetForPrompt(OpenEntityDataset):
    def _get_tags(self, d):
        tags = []
        for tag in d["y_str"]:
            tag_list = tag.split("_")
            tags += tag_list
        tags = list(set(tags))
        tags = [self.tag_mapping.get(t, None) for t in tags] # map to label words
        return tags

class OpenEntityGeneralDataset(EntityTypingDataset):

    def __load_data_from_file__(self, filepath):
        for file in filepath:
            if not os.path.exists(file):
                print(f"[ERROR] Data file {filepath} does not exist!")
                assert(0)
            with open(file, 'r', encoding='utf-8')as f:
                lines = f.readlines()
            data = json.loads(lines[0])
            for d in tqdm(data, desc="load"):
                tag = d["labels"]
                text = d["sent"]
                words = text.split(" ")
                start = len(text[:d["start"]].strip().split(" "))
                end = start + len(text[d["start"]:d["end"]].split(" "))
                label = [self.tag2id[t] for t in tag if t in self.tag2id]
                sample = OESample(words, tag, label, (start, end))
                # if sample.valid(self.max_length):
                self.samples.append(sample)
            
        # add <ENTITY> </ENTITY> 
        if self.highlight_entity:
            for sample in self.samples:
                sample.highlight(self.highlight_entity)
        
        print(f"{len(self.samples)} samples loaded.")
        
    def _get_filepath(self, datadir, mode):
        if mode == "test" or mode == "dev" or mode == "train":
            return [os.path.join(datadir, f'{mode}.json')]
        else:
            return [os.path.join(datadir, f'{mode}.json'), os.path.join(datadir, f"el_{mode}.json"), os.path.join(datadir, f"headword_{mode}.json")]

    
    def __sample_data__(self):
        if self.sample_num is None:
            return
        # for each type of entity, sample by sample_num
        sampler = FewshotSampler(self.sample_num, samples=self.samples)
        sampled_idx = sampler.__next__()
        self.samples = [self.samples[i] for i in sampled_idx]

OpenEntityGeneralDatasetForPrompt = OpenEntityGeneralDataset

def collate_fn(samples):
    batch_data = {'words':[], 'labels':[], 'entity_pos':[]}
    for sample in samples:
        batch_data['words'].append(sample.words)
        batch_data['labels'].append(sample.label)
        batch_data['entity_pos'].append(sample.pos)
    if isinstance(batch_data['labels'][0], int):
        batch_data['labels'] = torch.LongTensor(batch_data['labels'])
    batch_data['entity_pos'] = torch.LongTensor(batch_data['entity_pos'])
    return batch_data

def get_loader(dataset, batch_size, num_workers=4, collate_fn=collate_fn):
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn, drop_last=False)
    return data_loader
# %%
