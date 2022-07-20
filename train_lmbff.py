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
from openprompt_util.dataprocessor import BBNProcessor, OntoNoteProcessor, FewNerdProcessor
import argparse
import random
import numpy as np
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, T5TemplateGenerator
import torch
from openprompt import PromptForClassification
from util.metrics import get_metrics
from tqdm import tqdm
import os
from transformers import  AdamW, get_linear_schedule_with_warmup


use_cuda = True

processors = {"fewnerd": FewNerdProcessor, "ontonote": OntoNoteProcessor, "bbn": BBNProcessor}

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

def fit(model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler=None):
    best_score = 0.0
    for epoch in range(30):
        train_epoch(model, train_dataloader, loss_func, optimizer, scheduler=scheduler)
        score = evaluate(model, val_dataloader)
        if score > best_score:
            best_score = score
    return best_score


def train_epoch(model, train_dataloader, loss_func, optimizer, scheduler=None):
    global use_cuda
    model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

def evaluate(model, val_dataloader):
    global use_cuda
    model.eval()
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc

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

    # You can load the plm related things provided by openprompt simply by calling:
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name)

    # for example the verbalizer contains multiple label words in each class
    myverbalizer = ManualVerbalizer(tokenizer, classes=processor.labels).from_file(f"./openprompt_util/script/{args.data}/verbalizer.json")


    # prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    # if use_cuda:
    #     prompt_model=  prompt_model.cuda()

    # LMBFF: model for generating template
    template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm('t5', 't5-large')

    from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate
    import copy
    template = LMBFFTemplateGenerationTemplate(tokenizer=template_generate_tokenizer, verbalizer=myverbalizer, text='{"placeholder":"text_a"} {"mask"} {"meta":"labelword"}.')

    print("wrapped example for template generation:", template.wrap_one_example(train_dataset[0]))


    # ############################## LMBFF start #####################################
    print('performing auto_t...')

    if use_cuda:
        template_generate_model = template_generate_model.cuda()
    template_generator = T5TemplateGenerator(template_generate_model, template_generate_tokenizer, template_tokenizer_wrapper, myverbalizer, beam_width=100, target_number=1)


    dataloader = PromptDataLoader(train_dataset, template, tokenizer=template_generate_tokenizer, tokenizer_wrapper_class=template_tokenizer_wrapper, batch_size=len(train_dataset), decoder_max_length=128, max_seq_length=128, shuffle=False, teacher_forcing=False) # register all data at once
    for data in dataloader:
        if use_cuda:
            data = data.cuda()
        template_generator._register_buffer(data)
    
    template_filepath = f"generated_templates_{args.data}_{args.sample_num}_{args.seed}.txt"

    if not os.path.exists(template_filepath):
        template_generate_model.eval()
        print('generating...')

        template_texts = template_generator._get_templates()

        original_template = template.text
        template_texts = [template_generator.convert_template(template_text, original_template) for template_text in template_texts]
        # template_generator._show_template()
        # generate a number of candidate template text
        # print("generated templates:", template_texts)
        with open(template_filepath, "w")as f:
            f.writelines("\n".join(template_texts))
        print("generated templates saved")
    template_generator.release_memory()

    template_texts = [t.strip() for t in open(template_filepath).readlines()]

    
    # iterate over each candidate and select the best one
    best_metrics = 0.0
    best_template_text = None
    for template_text in tqdm(template_texts):
        template = ManualTemplate(tokenizer, template_text)

        train_dataloader = PromptDataLoader(train_dataset, template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size, decoder_max_length=128, max_seq_length=args.max_length, shuffle=True, teacher_forcing=False)
        valid_dataloader = PromptDataLoader(dev_dataset, template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, batch_size=args.val_batch_size, decoder_max_length=128, max_seq_length=args.max_length, shuffle=False, teacher_forcing=False)

        model = PromptForClassification(copy.deepcopy(plm), template, myverbalizer)

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        global_train_iter = len(train_dataloader) * args.epoch 
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=global_train_iter)
        if use_cuda:
            model = model.cuda()
        score = fit(model, train_dataloader, valid_dataloader, loss_func, optimizer, scheduler=scheduler)

        if score > best_metrics:
            print('best score:', score)
            print('template:', template_text)
            best_metrics = score
            best_template_text = template_text
    # use the best template
    mytemplate = ManualTemplate(tokenizer, text=best_template_text)
    print("final best templates:", best_template_text)
    with open(template_filepath[:-4] + "_best.txt", "w")as f:
        f.writelines(best_template_text)

    # ############################## LMBFF done #####################################

    train_dataloader = PromptDataLoader(train_dataset, mytemplate, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size, decoder_max_length=3, max_seq_length=args.max_length, shuffle=True, teacher_forcing=False)
    dev_dataloader = PromptDataLoader(dev_dataset, mytemplate, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size, decoder_max_length=3, max_seq_length=args.max_length, shuffle=False, teacher_forcing=False)
    test_dataloader = PromptDataLoader(test_dataset, mytemplate, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size, decoder_max_length=3, max_seq_length=args.max_length, shuffle=False, teacher_forcing=False)

    prompt_model = PromptForClassification(plm, mytemplate, myverbalizer)
    if use_cuda:
        prompt_model = prompt_model.cuda()


    # Now the training is standard
    loss_func = torch.nn.CrossEntropyLoss()
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
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step %1000 ==1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
                global_step += 1

                if global_step % args.val_step == 0:
                    allpreds = []
                    alllabels = []
                    for step, inputs in enumerate(dev_dataloader):
                        if use_cuda:
                            inputs = inputs.cuda()
                        logits = prompt_model(inputs)
                        labels = inputs['label']
                        alllabels.extend(labels.cpu().tolist())
                        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

                        if step == 200:
                            break

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
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    idx2oritag = dict(zip(range(len(processor.labels)), processor.labels))
    oritag2idx = dict(zip(processor.labels, range(len(processor.labels))))

    if 'ontonote' in args.data:
        alllabels = torch.LongTensor(alllabels)
        allpreds = torch.LongTensor(allpreds)
        allpreds = allpreds[alllabels != oritag2idx['/other']]
        alllabels = alllabels[alllabels != oritag2idx['/other']]
        alllabels = alllabels.numpy().tolist()
        allpreds = allpreds.numpy().tolist()
    acc, micro, macro = get_metrics(alllabels, allpreds, idx2oritag, isfewnerd=args.data=="fewnerd")

    # acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print(f"TEST RESULT: \nacc: {acc}\nmicro: {micro}\nmacro:{macro}", )

    with open("result_supervised_lmbff.txt", "a+")as f:
        f.writelines(f"model: {args.model}\nsample_num: {args.sample_num}\ndata:{args.data}\n")
        f.writelines(f"TEST RESULT: \nacc: {acc}\nmicro: {micro}\nmacro:{macro}\n\n")



if __name__ == "__main__":
    main()