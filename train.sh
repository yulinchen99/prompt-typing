data='fewnerd' # "ontonotes", "openentity", "bbn"
echo $data

# use soft-prompt
prompt='soft'
# use hard-prompt (main experiments)
# prompt='hard3'

# baseline BERT-CLS Finetuning
python -u train.py --model baseline --model_name bert-base-cased --data $data --lr 2e-5 --usecls --epoch 500 --batch_size 32 --val_batch_size 64 --val_step 2000 --log_step 1000

# prompt typing 
# 1 shot
python -u train.py --model maskedlm --model_name bert-base-cased --data $data --prompt $prompt  --lr 5e-5 --sample_num 1

# 8 shot
python -u train.py --model maskedlm --model_name bert-base-cased --data $data --prompt $prompt  --lr 5e-5 --sample_num 8
