
python -u train_op_openentity.py --epoch 20 --val_step 200 --lr 2e-5 --data openentity-general --model roberta --model_name roberta-large --batch_size 16
python -u train_op_openentity.py --epoch 20 --val_step 200 --lr 2e-5 --data openentity-general --model gpt2 --model_name gpt2 --batch_size 16
python -u train_op_openentity.py --epoch 20 --val_step 200 --lr 2e-5 --data openentity-general --model T5 --model_name T5-base --batch_size 16

