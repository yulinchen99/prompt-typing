for data in "fewnerd" "ontonote" "bbn" 
do
for seed in 0
do
for n in 4 8 16
do
echo $data
# highlight_entity='[ENT]-[ENT]'
# soft-prompt
# prompt='hard3'
# hard-prompt
#prompt='hard1'


python -u train_lmbff.py --model bert --model_name bert-base-cased --data $data --lr 5e-5 --epoch 30 --batch_size 32 --val_batch_size 64 --val_step 20 --log_step 20 --sample_num $n --seed $seed
done
done
done