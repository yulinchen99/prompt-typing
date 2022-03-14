# In this scripts, you will learn 
# 1. how to use integrate huggingface datasets utilities into openprompt to
#  enable prompt learning in diverse datasets.
# 2. How to instantiate a template using a template language
# 3. How does the template wrap the input example into a templated one.
# 4. How do we hide the PLM tokenization details behind and provide a simple tokenization
# 5. How do construct a verbalizer using one/many label words
# 5. How to train the prompt like a traditional Pretrained Model.


from openprompt.data_utils import InputExample, data_sampler
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
# from openprompt.data_utils.typing_dataset import FewNERDProcessor
from openprompt_util.dataprocessor import BBNProcessor, OntoNoteProcessor, FewNerdProcessor, OpenEntityProcessor
import argparse
import random
import numpy as np
# from openprompt import PromptDataLoader
from openprompt_util.dataloader_openentity import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
import torch
from openprompt import PromptForClassification
from util.metrics import get_metrics
from tqdm import tqdm
import os
from transformers import  AdamW, get_linear_schedule_with_warmup
from util.util import MultiLabelLoss, get_output_index
from util.metrics import get_openentity_metrics_for_prompt



processors = {"openentity": OpenEntityProcessor}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def to_cuda(data):
    for item in data:
        if isinstance(data[item], torch.LongTensor):
            data[item] = data[item].cuda()

def main():
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default='fewnerd', help='ontonote, fewnerd or bbn')
    parser.add_argument('--model', type=str, default='t5', help='model type')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--embed_lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=200)
    parser.add_argument('--grad_accum_step', type=int, default=1)
    parser.add_argument('--warmup_step', type=int, default=100)
    parser.add_argument('--val_step', type=int, default=2000, help='val every x steps of training')
    parser.add_argument('--log_step', type=int, default=2000, help='log every x steps of training')

    parser.add_argument('--val_iter', type=int, default=None, help='val iter')
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--result_save_dir', type=str, default='result')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--sample_num', type=int, default=None, help='default training on all samples, set a number to indicate how many samples in each type are sampled as training set')
    parser.add_argument('--calibrate', action='store_true', default=False)
    parser.add_argument('--save_result', action='store_true', default=False)




    # for soft prompt only
    parser.add_argument('--prompt', type=str, default='soft', help='soft or hard')
    #parser.add_argument('--dual_optim', action='store_true', default=False, help='set True if separate learning rate in maskedlm p-prompt setting is desired')
    parser.add_argument('--dropout', type=float, default=0.1)

    # for baseline only
    parser.add_argument('--usecls', action='store_true', default=False)
    parser.add_argument('--highlight_entity', type=str, default=None, help='for baseline model, highlight tokens around entity')
    parser.add_argument('--loss', type=str, default='cross', help='cross or partial')


    args = parser.parse_args()
    # set random seed
    set_seed(args.seed)

    processor = processors[args.data]()
    train_dataset = processor.get_train_examples(f"./data/{args.data}/")
    dev_dataset = processor.get_dev_examples(f"./data/{args.data}/")
    test_dataset = processor.get_test_examples(f"./data/{args.data}/")

    if args.sample_num is not None:
        sampler = data_sampler.FewShotSampler(num_examples_per_label=args.sample_num)
        train_dataset = sampler(train_dataset, seed=args.seed)
    
    if args.sample_num is not None and len(train_dataset) < len(dev_dataset):
        indices = torch.randperm(len(dev_dataset), generator=torch.Generator().manual_seed(0)).tolist()
        dev_dataset = [dev_dataset[i] for i in indices[:len(train_dataset)]]
        # dev_dataset, _ = torch.utils.data.random_split(dev_dataset, [len(train_dataset), len(dev_dataset)-len(train_dataset)], generator=torch.Generator().manual_seed(0))

    # You can load the plm related things provided by openprompt simply by calling:
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name)

    # Constructing Template
    # A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
    template_text = '{"placeholder":"text_a"} In this sentence, {"meta": "entity"} is a {"mask"}.'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)


    # We provide a `PromptDataLoader` class to help you do all the above matters and wrap them into an `torch.DataLoader` style iterator.

    train_dataloader = PromptDataLoader(dataset=train_dataset, template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False)
    dev_dataloader = PromptDataLoader(dataset=dev_dataset, template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False)
    test_dataloader = PromptDataLoader(dataset=test_dataset, template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False)
    # next(iter(train_dataloader))


    # Define the verbalizer
    # In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:



    # for example the verbalizer contains multiple label words in each class
    myverbalizer = ManualVerbalizer(tokenizer, classes=processor.labels).from_file(f"./openprompt_util/script/{args.data}/verbalizer.json")

    # print(myverbalizer.label_words_ids)
    # logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and 
    # print(myverbalizer.process_logits(logits)) # see what the verbalizer do


    # Although you can manually combine the plm, template, verbalizer together, we provide a pipeline 
    # model which take the batched data from the PromptDataLoader and produce a class-wise logits


    use_cuda = True
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model=  prompt_model.cuda()

    # Now the training is standard
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = MultiLabelLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    best_acc = 0.0
    global_step = 0
    if not args.test_only:
        for epoch in range(args.epoch):
            tot_loss = 0 
            for step, inputs in tqdm(enumerate(train_dataloader)):
                if use_cuda:
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            inputs[k] = inputs[k].cuda()
                    # inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs["label"]
                # print(labels)
                loss = loss_func(logits, labels, model_type="maskedlm")
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step %100 ==1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
                global_step += 1

                if global_step % args.val_step == 0:
                    allpreds = []
                    alllabels = []
                    for step, inputs in enumerate(dev_dataloader):
                        if use_cuda:
                            for k in inputs:
                                if isinstance(inputs[k], torch.Tensor):
                                    inputs[k] = inputs[k].cuda()
                        logits = prompt_model(inputs)
                        labels = inputs['label']
                        pred = get_output_index(logits)
                        alllabels.extend([l.cpu().tolist() for l in labels])
                        allpreds.append(pred)

                    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
                    print("validation acc:", acc)
                    if acc > best_acc:
                        print("best checkpoint!")
                        best_acc = acc
                        plm.save_pretrained(f"output/{args.model}-{args.data}-{args.sample_num}")
    
    if not args.test_only and os.path.exists(f"output/{args.model}-{args.data}-{args.sample_num}"):
        plm = plm.from_pretrained(f"output/{args.model}-{args.data}-{args.sample_num}")
        # print(plm)
        prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        if use_cuda:
            prompt_model = prompt_model.cuda()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        pred = get_output_index(logits)
        alllabels.extend([l.cpu().tolist() for l in labels])
        allpreds.extend(pred)

    idx2tag = dict(zip(range(len(processor.labels)), processor.labels))
    # tag2idx = dict(zip(range(len(processor.labels)), processor.labels))
    with open("./data/openentity/types.txt")as f:
        lines = f.readlines()
        ori_tag_list = [line.strip() for line in lines]
    idx2oritag = dict(zip(range(len(ori_tag_list)), ori_tag_list))
    oritag2idx = dict(zip(ori_tag_list, range(len(ori_tag_list))))
    acc, micro, macro = get_openentity_metrics_for_prompt(alllabels, allpreds, idx2tag, idx2oritag, oritag2idx, ori_tag_list)

    # acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print(f"TEST RESULT: \nacc: {acc}\nmicro: {micro}\nmacro:{macro}", )

    with open("result_openentity.txt", "a+")as f:
        f.writelines(f"model: {args.model}\nsample_num: {args.sample_num}\ndata:{args.data}\nzero-shot: {args.test_only}\n")
        f.writelines(f"TEST RESULT: \nacc: {acc}\nmicro: {micro}\nmacro:{macro}\n\n")



if __name__ == "__main__":
    main()