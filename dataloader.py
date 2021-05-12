import torch
import torch.utils.data as data
import os
import numpy as np
import random
import copy
from transformers import BertTokenizer, AutoConfig, RobertaConfig, BertConfig, RobertaTokenizer
# from word_encoder import BERTWordEncoder

class Sample:
    def __init__(self, filelines, max_length, mask, tag_mapping):
        filelines = [line.split('\t') for line in filelines]
        self.words, self.tags = zip(*filelines)
        self.entity_positions = {}
        self.max_length = max_length
        self.mask = mask
        self.map_tags(tag_mapping)
        self.__get_entities__()

    def __get_entities__(self):
        current_tag = None
        entity_pos = []
        for idx, tag in enumerate(list(self.tags)+['O']):
            if tag == current_tag:
                continue
            else:
                if current_tag is not None and current_tag != 'O':
                    entity_pos.append(idx)
                    assert len(entity_pos) == 2, print(entity_pos, self.words, self.tags)
                    self.entity_positions[tuple(entity_pos)]=current_tag
                    entity_pos = []
                if tag != 'O':
                    entity_pos.append(idx)
                current_tag = tag

    def get_samples(self, sample_num, max_length):
        tmp_candidate_items = copy.deepcopy(self.candidate_items)
        if len(self.candidate_items) > sample_num:
            tmp_candidate_items = random.sample(tmp_candidate_items, sample_num)
        word_input, word_labels = self.__get_word_mask_samples__(tmp_candidate_items)
        tag_input, tag_labels = self.__get_tag_mask_samples__(tmp_candidate_items)
        d = {'word_input':word_input, 'word_labels':word_labels, 'tag_input':tag_input, 'tag_labels':tag_labels}
        return d

    def __get_tag_mask_samples__(self, sampled_entities):
        inputs = []
        labels = []
        for pos, tag in sampled_entities:
            masked = list(copy.deepcopy(self.words[:self.max_length]))
            masked += masked[pos[0]:pos[1]] + ['is', self.mask]
            inputs.append(masked)
            labels.append(tag)
        return inputs, labels

    def __get_word_mask_samples__(self, sampled_entities):
        inputs = []
        labels = []
        for pos, _ in sampled_entities:
            masked = list(copy.deepcopy(self.words[:self.max_length]))
            masked[pos[0]:pos[1]] = [self.mask]*(pos[1]-pos[0])
            inputs.append(masked)
            labels.append(list(self.words[:self.max_length]))
        return inputs, labels

    def map_tags(self, tag_mapping):
        self.tags = [tag_mapping.get(tag, 'O') for tag in self.tags]

    def empty(self):
        self.candidate_items = [item for item in self.entity_positions.items() if item[0][1] < self.max_length]
        if not self.candidate_items:
            return True
        else:
            return False

    def __str__(self):
        newlines = zip(self.words, self.tags)
        text = '\n'.join(['\t'.join(line) for line in newlines])
        return f'entities: {self.entity_positions}, text: {text}'


class OpenNERDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, filepath, tokenizer, max_length, sample_num, tag2idx, tag_mapping):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.samples = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        # number of entities per sentence
        self.sample_num = sample_num
        self.tag2id = tag2idx
        self.tag_mapping = tag_mapping
        self.__load_data_from_file__(filepath)
    
    def __load_data_from_file__(self, filepath):
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                samplelines.append(line)
            else:
                if not samplelines:
                    continue
                sample = Sample(samplelines, self.max_length, self.tokenizer.mask_token, self.tag_mapping)
                if not sample.empty():
                    self.samples.append(sample)
                samplelines = []
                index += 1

    def tokenize(self, words):
        output = self.tokenizer(words, is_split_into_words=True,return_attention_mask=True)
        return output.input_ids, output.attention_mask
    
    def __getitem__(self, index):
        # get raw data
        data = self.samples[index].get_samples(self.sample_num, self.max_length)
        # tokenize
        '''
        if not data:
            return {'word_input':[], 'word_labels':[], 'tag_input':[], 'tag_labels':[], 'word_mask':[], 'tag_mask':[]}
        '''
        word_input, word_mask = self.tokenize(data['word_input'])
        tag_input, tag_mask = self.tokenize(data['tag_input'])
        word_labels, _ = self.tokenize(data['word_labels'])
        tag_labels = [self.tag2id[tag] for tag in data['tag_labels']]
        newdata = {'word_input':word_input, 'word_labels':word_labels, 'tag_input':tag_input, 'tag_labels':tag_labels, 'word_mask':word_mask, 'tag_mask':tag_mask}
        return newdata

    def __len__(self):
        return len(self.samples)


def get_tokenizer(model_name):
    tokenizer = None
    config = AutoConfig.from_pretrained(model_name)
    if isinstance(config, RobertaConfig):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif isinstance(config, BertConfig):
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        print('unsupported model name')
    return tokenizer

def collate_fn(data):
    max_length = np.max([len(d) for dat in data for d in dat['tag_input']])
    def pad(seq):
        while len(seq) < max_length:
            seq.append(0)
        return seq
    batch_data = {'word_input':[], 'word_labels':[], 'tag_input':[], 'tag_labels':[], 'word_mask':[], 'tag_mask':[]}
    # padding
    #print(max_length)
    for d in data:
        for name in d:
            if name == 'tag_labels':
                for seq in d[name]:
                    batch_data[name].append(seq)
            else:
                for seq in d[name]:
                    batch_data[name].append(pad(seq))
    for name in batch_data:
        batch_data[name] = torch.LongTensor(batch_data[name])
    # assert batch_data['word_input'].shape == batch_data['word_mask'].shape, print(batch_data['word_input'].shape, batch_data['word_mask'].shape)
    return batch_data

def get_loader(filepath, tokenizer, batch_size, max_length, sample_num, tag_list, tag_mapping, num_workers=8, collate_fn=collate_fn):
    dataset = OpenNERDataset(filepath, tokenizer, max_length, sample_num, tag_list, tag_mapping)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader
