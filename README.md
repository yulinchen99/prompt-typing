# Prompt Typing
Code and data for EMNLP 2022 paper Prompt-Learning for Fine-Grained Entity Typing.

This repository contains code for experiments on BBN, Few-NERD, OntoNotes and OpenEntity with BERT-base-cased.

## Project Structure
```
.
├── data
│   ├── bbn
│   ├── fewnerd
│   ├── ontonote
│   ├── openentity
│   ├── openentity-general  # not used in our experiments
├── distant-pretraining # code for semi-supervised learning in zero-shot setting
├── model
│   ├── baseline.py # model for vanilla model fine-tuning
│   ├── maskedlm.py # model for prompt typing
├── util # util function, metrics and dataloader
│   ├── data_loader.py
│   ├── fewshotsampler.py
│   ├── metrics.py
│   ├── util.py
├── train.py # main access
└── README.md
```

## Required Packages
- pytorch
- transformers
- sklearn
- pandas
- tqdm

## How to Run
### Explanation of main command arguments
```
python -u train.py \
--model maskedlm \ # training mode, "baseline" for vanilla FT, "maskedlm" for prompt typing
--model_name bert-base-cased \ # pretrained model path
--data fewnerd \ # fewnerd, bbt, ontonote, or openentity
--prompt hard  \ # type of prompts, see details below
--lr 5e-5 \
--sample_num 1 \ # training data shot number
--seed 0
```

### Run with multiple settings
- Run 1-shot 
```
python -u train.py --model maskedlm --model_name bert-base-cased --data fewnerd --prompt hard  --lr 5e-5 --sample_num 1
```

- Run full-supervised setting
```
python -u train.py --model maskedlm --model_name bert-base-cased --data fewnerd --prompt hard  --lr 5e-5
```


- Run zero-shot setting
    - Download semi-supervised pretrained model checkpoint
    ```
    cd distant-pretraining/result
    bash download.sh
    cd ../../
    ```
    - Run the test
    ```
    python -u train.py --model maskedlm --model_name bert-base-cased --data fewnerd --prompt hard  --lr 5e-5 --test_only --load_ckpt distant-pretraining/result/best-checkpoint/5000
    ```

### Run with various template types
Specify `--prompt` arguments to denote which template to be used. Below is a list of supported values and corresponding template format
- `hard1`: `<text> <entity> is <mask>`
- `hard2`: `<text> <entity> is a <mask>`
- `hard3`: `<text> Inthis sentence, <entity> is a <mask>`
- `soft`: `<text> [P] <entity> [P1] [P2] <mask>`
- `soft1`: `<text> [P] <entity> [P1] [P2] [P3] <mask>`
- `soft2`: `<text> [P] <entity> [P1] [P2] [P3] [P4] <mask>`
- `soft3`: `<text> [P] <entity> [P1] [P2] [P3] [P4] <mask>`
