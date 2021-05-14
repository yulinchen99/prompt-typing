# %%
import pandas as pd
def load_tag_mapping(filepath):
    df=pd.read_csv(filepath, sep='\t', header=None)
    tags = []
    for i in df[4]:
        tags += i.split(' ')

    tags = list(set(tags))        
    newtags = []
    for tag in tags:
        tag = tag.strip('/').lower()
        tag_split = tag.split('/')
        if len(tag_split) == 1:
            newtags.append(tag_split[0])
        else:
            if len(tag_split) > 1:
                newtag = '/'.join(tag_split[-2:]).replace('_', '/').replace('/of', '').replace('/thing', '').replace('/and', '').replace('other', '').replace('//', '/').strip('/')
            else:
                newtag = '/'.join(tag_split).replace('_', '/').replace('/of', '').replace('/thing', '').replace('/and', '').replace('//', '/').strip('/')
            newtags.append(newtag)
    return dict(zip(tags, newtags))

# %%
from collections import Counter
d = load_tag_mapping('OntoNotes-89/train_clean.tsv')
print(d)
print(len(d))
cnt = Counter(d.values())
print(cnt)
# %%
