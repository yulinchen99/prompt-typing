import json
def load_tag_mapping(filepath):
    with open('tag_list.txt')as f:
        lines = f.readlines()
        ori_tag_list = [line.strip() for line in lines]
    with open(filepath)as f:
        lines = f.readlines()
        new_tag_list = [line.strip().lower() for line in lines]
    tag_mapping = dict(zip(ori_tag_list, new_tag_list))
    print(tag_mapping)
    return tag_mapping

def get_tag2inputid(tokenizer, tags):
    splitted_tags = [list(set(tag.split('/'))) for tag in tags]
    d ={}
    for i, splitted_tag in enumerate(splitted_tags):
        d[tags[i]] = tokenizer.convert_tokens_to_ids(splitted_tag)
    return d

class ResultLog:
    '''
    create json log
    {
        model_name:
        train_file:
        val_file:
        test_file:
        seed:
        model_save_dir:
        result: {
            0: {
                train_loss: 
                train_acc: 
                val_acc: 
            }
            ...
        }
    }
    '''
    def __init__(self, args, save_path):
        d = {'model_name':args.model_name, 'train_file':args.train_file, 'val_file':args.val_file, 'test_file': args.test_file, 'seed':args.seed, 'model_save_dir':args.model_save_dir, 'result':{}}
        with open(save_path, 'w')as f:
            f.write(json.dumps(d))
        self.save_path = save_path

    def update(self, epoch, data):
        '''
        data = {train_acc: , test_acc: , train_loss: }
        '''
        with open(self.save_path, 'r+')as f:
            d = json.load(f)
            f.seek(0)
            f.truncate()
            d['result'][epoch] = data
            f.write(json.dumps(d))