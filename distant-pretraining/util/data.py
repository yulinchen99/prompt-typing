import torch.utils.data as data
import json
from tqdm import tqdm
import torch
class Pair:
    def __init__(self, pair_data):
        self.sent1 = pair_data[0]['tokens']
        self.sent2 = pair_data[1]['tokens']
        self.pos1 = pair_data[0]['pos']
        self.pos2 = pair_data[1]['pos']
        self.entity_name = pair_data[2]
        self.label = pair_data[3]

    def __str__(self):
        return 'sentences:' + '\n'
        + ' '.join(self.sent1) + '\n'
        + ' '.join(self.sent2) + '\n'
        + 'entity:' + self.entity_name + '\n'
        + 'label:' + str(self.label)

class PairDataset(data.Dataset):
    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.load_data()

def load_data(sample_path):
    def load_fn(line):
        return Pair(json.loads(line.strip()))
    # return List[Pair]
    datalist = []
    with open(sample_path)as f:
        datalist = json.loads(f.read().strip())
    #return list(map(load_fn, lines))
    return datalist

def mycollate_fn(pair_list):
    sent1 = []
    sent2 = []
    pos1 = []
    pos2 = []
    entity_name = []
    labels = []
    for pair in pair_list:
        #print(pair)
        '''
        sent1.append(pair.sent1)
        sent2.append(pair.sent2)
        pos1.append(pair.pos1)
        pos2.append(pair.pos2)
        entity_name.append(pair.entity_name)
        labels.append(pair.label)
        '''
        sent1.append(pair[0]['tokens'])
        sent2.append(pair[1]['tokens'])
        pos1.append(pair[0]['pos'])
        pos2.append(pair[1]['pos'])
        entity_name.append(pair[2])
        labels.append(pair[3])
    return {
        'sent1': sent1,
        'sent2': sent2,
        'pos1': pos1,
        'pos2': pos2,
        'entity_name': entity_name,
    }, torch.LongTensor(labels)

def get_loader(pair_list, batch_size, num_workers=8, collate_fn=mycollate_fn):
    data_loader = data.DataLoader(dataset=pair_list,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn, drop_last=True)
    return data_loader

