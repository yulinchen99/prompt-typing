data='ontonote'
highlight_entity='<ENTITY>-</ENTITY>'
# soft-prompt
prompt='soft'
# hard-prompt
#prompt='hard'

# baseline BERT-CLS
python -u train.py --model baseline --model_name roberta-base --data $data --highlight_entity $highlight_entity --lr 1e-5

# baseline 10% train data
python -u train.py --model baseline --model_name roberta-base --data $data --highlight_entity $highlight_entity --lr 1e-5 --sample_rate 0.1

# prompt model
python -u train.py --model maskedlm --model_name roberta-base --data $data --prompt $prompt  --lr 1e-5

# 10% train data
python -u train.py --model maskedlm --model_name roberta-base --data $data --prompt $prompt  --lr 1e-5 --sample_rate 0.1

# test only
# python -u train.py --model maskedlm --model_name roberta-base --data $data --prompt $prompt --test_only --load_ckpt ...