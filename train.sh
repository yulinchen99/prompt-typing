data='ontonote'
highlight_entity='<ENTITY>-</ENTITY>'
# p-prompt
prompt='[P]-[P1]-[P2]'
# is-prompt
#prompt='is'

# baseline BERT-CLS
python -u train.py --model baseline --model_name roberta-base --data $data --highlight_entity $highlight_entity --lr 1e-5

# Ours one optimizer
python -u train.py --model maskedlm --model_name roberta-base --data $data --prompt $prompt  --lr 1e-5

# Ours separate optimizer
python -u train.py --model maskedlm --model_name roberta-base --data $data --prompt $prompt --dual_optim --lr 1e-5 --embed_lr 1e-4

# test only
# python -u train.py --model maskedlm --model_name roberta-base --data $data --prompt $prompt --test_only