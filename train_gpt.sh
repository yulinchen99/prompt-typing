
for data in "fewnerd" "ontonote" "bbn"
do
# python -u train_op.py --sample_num 1 --epoch 50 --val_step 10 --lr 5e-5 --data $data --model gpt2 --model_name gpt2
# python -u train_op.py --sample_num 2 --epoch 50 --val_step 10 --lr 5e-5 --data $data --model gpt2 --model_name gpt2
# python -u train_op.py --sample_num 4 --epoch 50 --val_step 20 --lr 5e-5 --data $data --model gpt2 --model_name gpt2
# python -u train_op.py --sample_num 8 --epoch 50 --val_step 20 --lr 5e-5 --data $data --model gpt2 --model_name gpt2
# python -u train_op.py --sample_num 16 --epoch 50 --val_step 40 --lr 5e-5 --data $data --model gpt2 --model_name gpt2
python -u train_op.py --epoch 5 --val_step 2000 --lr 5e-5 --data $data --model gpt2 --model_name gpt2 --batch_size 64 --val_batch_size 128
# python -u train_op.py --epoch 10 --val_step 2000 --lr 5e-5 --data $data --model gpt2 --model_name gpt2 --test_only
done