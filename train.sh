data='openentity'
echo $data
# highlight_entity='<ENTITY>-</ENTITY>'
# soft-prompt
prompt='soft1'
# hard-prompt
#prompt='hard1'

# baseline BERT-CLS
# python -u train.py --model baseline --model_name bert-base-cased --data $data --lr 5e-5 --usecls

# baseline 1 shot train data
# python -u train.py --model baseline --model_name bert-base-cased --data $data --highlight_entity $highlight_entity --lr 5e-5 --sample_num 1 --usecls
python -u train.py --model baseline --model_name bert-base-cased --data $data --lr 5e-5 --usecls --epoch 20 --val_step 2000


# # prompt model
# python -u train.py --model maskedlm --model_name bert-base-cased --data $data --prompt $prompt  --lr 5e-5

# # 1 shot train data
# python -u train.py --model maskedlm --model_name bert-base-cased --data $data --prompt $prompt  --lr 5e-5 --sample_num 1

