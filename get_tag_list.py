# %%
def get_tag_list(datafile):
    with open(datafile)as f:
        lines = f.readlines()
        tag_list = list(set([line.split('\t')[1].strip() for line in lines if line.strip()]))
    return tag_list
# %%
tag_list = get_type_list('../model/data/data/mydata/test-inter-new.txt') + get_type_list('../model/data/data/mydata/train-inter-new.txt') + get_type_list('../model/data/data/mydata/val-inter-new.txt')
tag_list = list(set(tag_list))
tag_list.remove('O')
# %%
with open('tag_list.txt', 'w')as f:
    f.writelines('\n'.join(tag_list))

# %%
