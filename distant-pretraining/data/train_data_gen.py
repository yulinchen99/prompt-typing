import json
import random
from tqdm import tqdm
random.seed(0)
class DataGenerator:
    def __init__(self, inputfile, outputfile):
        with open(inputfile)as f:
            data = f.read()
        self.entity_dict = json.loads(data)
        self.entity_list = list(self.entity_dict.keys())
        self.outputfile = outputfile

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

    def generate_negative_sample(self, pair_num_per_sample=1):
        neg_sample = []
        for e in tqdm(self.entity_dict):
            for sample1 in self.entity_dict[e]:
                for _ in range(pair_num_per_sample):
                    e2 = random.choice(self.entity_list)
                    while e2 == e:
                        e2 = random.choice(self.entity_list)
                    sample2 = random.choice(self.entity_dict[e2])
                    neg_sample.append([sample1, sample2, e, 0])
        return neg_sample

    def generate_sample(self, num=1, sent_per_entity = 10):
        self.sample_sentences(sample_num=sent_per_entity)
        pos_sample = self.generate_positive_sample(pair_num_per_sample=num)
        neg_sample = self.generate_negative_sample(pair_num_per_sample=num)
        sample = pos_sample + neg_sample
        with open(self.outputfile,  'w')as f:
            f.writelines(json.dumps(sample))


if __name__ == '__main__':
    generator = DataGenerator('./distant_entity.json', 'samples.json')
    generator.generate_sample()




