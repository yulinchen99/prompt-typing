import json
import random
from tqdm import tqdm
import os
random.seed(0)
class DataGenerator:
    def __init__(self, inputfile, outputfile, dictdir):
        with open(inputfile)as f:
            data = f.read()
        self.entity_dict = json.loads(data)
        self.entity_list = list(self.entity_dict.keys())
        self.outputfile = outputfile
        self.dictdir = dictdir

    def sample_sentences(self, sample_num=10):
        for e in tqdm(self.entity_dict):
            if len(self.entity_dict[e]) > 10:
                self.entity_dict[e] = random.sample(self.entity_dict[e], sample_num)
    
    def generate_positive_sample(self, pair_num_per_sample=1):
        pos_sample = []
        for e in tqdm(self.entity_dict):
            if len(self.entity_dict[e]) > 1:
                for sample1 in self.entity_dict[e]:
                    for _ in range(pair_num_per_sample):
                        sample2 = random.choice(self.entity_dict[e])
                        i = 0
                        while ''.join(sample2['tokens']) == ''.join(sample1['tokens']):
                            sample2 = random.choice(self.entity_dict[e])
                            i += 1
                            #print('resample')
                            if i == 100:
                                print(self.entity_dict[e])
                        pos_sample.append([sample1, sample2, e, 1])
        return pos_sample
    
    def load_entity2type(self):
        filenames = os.listdir(self.dictdir)
        d = {}
        for filename in filenames:
            e_type = filename.split('-')[0]
            with open(os.path.join(self.dictdir, filename))as f:
                entities = f.readlines()
                entities = [e.strip().lower() for e in entities]
                for e in entities:
                    d[e] = e_type
        self.entity2type = d

    def same_type(self, e1, e2):
        type1 = self.entity2type.get(e1, False)
        type2 = self.entity2type.get(e2, False)
        return type1 and type2 and type1 == type2

    def filter_entity_dict(self):
        entity_dict = {}
        for e in self.entity_dict:
            if self.entity2type.get(e.lower(), False):
                entity_dict[e.lower()] = self.entity_dict[e]
        return entity_dict


    def generate_negative_sample(self, pair_num_per_sample=1):
        self.load_entity2type()
        print(len(self.entity_dict))
        entity_dict = self.filter_entity_dict()
        entity_list = list(entity_dict.keys())
        print(len(entity_list))
        neg_sample = []
        for e in tqdm(entity_dict):
            for sample1 in entity_dict[e]:
                for _ in range(pair_num_per_sample):
                    e2 = random.choice(entity_list)
                    while self.same_type(e, e2):
                        e2 = random.choice(entity_list)
                    sample2 = random.choice(entity_dict[e2])
                    neg_sample.append([sample1, sample2, e, 0])
        return neg_sample

    def generate_sample(self, num=1, sent_per_entity = 10):
        print('sample sentences')
        self.sample_sentences(sample_num=sent_per_entity)
        print('generating pos samples')
        pos_sample = self.generate_positive_sample(pair_num_per_sample=num)
        print('num of pos', len(pos_sample))
        print('generating neg samples')
        neg_sample = self.generate_negative_sample(pair_num_per_sample=2)
        print('num of neg', len(neg_sample))
        sample = pos_sample + neg_sample
        print(len(sample))
        with open(self.outputfile,  'w')as f:
            f.writelines(json.dumps(sample))


if __name__ == '__main__':
    generator = DataGenerator('./distant_entity.json', 'samples.json', './dict')
    generator.generate_sample()