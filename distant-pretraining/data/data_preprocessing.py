# %%
import json
from tqdm import tqdm
class Entity:
    def __init__(self, entity_data, first_only=True):
        self.id = entity_data['id']
        self.name = entity_data['name']
        if first_only:
            self.pos = entity_data['pos'][0]
        else:
            self.pos = entity_data['pos']

    def __eq__(self, other):
        return self.name.lower() == other.name.lower()
    
    def __hash__(self):
        return hash(self.name.lower())

class Sentence:
    def __init__(self, sent_data):
        self.tokens = sent_data['tokens']
        self.entities = list(set([Entity(sent_data['h']), Entity(sent_data['t'])]))
        self.r = sent_data['r']

    def __eq__(self, other):
        return hash(''.join(self.tokens)) == hash(''.join(other.tokens))

    def __hash__(self):
        return hash(''.join(self.tokens))

    def update_entity(self, other):
        self.entities = list(set(self.entities).union(set(other.entities)))

# %%
if __name__ == '__main__':
    s_dict = {}
    with open('distant.json', encoding='utf-8')as f:
        txt = f.read()
    data = json.loads(txt.strip())
    for d in tqdm(data):
        for sent in data[d]:
            s = Sentence(sent)
            if s in s_dict:
                s_dict[s].update_entity(s)
            else:
                s_dict[s] = s
# %%
    all_entities = []
    for s in s_dict:
        all_entities += s_dict[s].entities
    all_entities = list(set(all_entities))

    entity_distant_dict = {e.name.lower():[] for e in all_entities}
    for s in tqdm(s_dict):
        for e in s_dict[s].entities:
            entity_distant_dict[e.name.lower()].append({
                'tokens': s.tokens,
                'pos': e.pos
            })
# %%
    with open('distant_entity.json', 'w')as f:
        f.write(json.dumps(entity_distant_dict))

# %%
