export CUDA_VISIBLE_DEVICES=0

model_name=LSTM

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali.csv \
  --data_train_path yali_train.csv \
  --data_vali_path yali_vali.csv \
  --data_test_path yali_test.csv \
  --model_id yali_10_10 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 10 \
  --label_len 5 \
  --pred_len 10 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali.csv \
  --data_train_path yali_train.csv \
  --data_vali_path yali_vali.csv \
  --data_test_path yali_test.csv \
  --model_id yali_10_15 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 10 \
  --label_len 5 \
  --pred_len 15 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali.csv \
  --data_train_path yali_train.csv \
  --data_vali_path yali_vali.csv \
  --data_test_path yali_test.csv \
  --model_id yali_10_20 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 10 \
  --label_len 5 \
  --pred_len 20 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali.csv \
  --data_train_path yali_train.csv \
  --data_vali_path yali_vali.csv \
  --data_test_path yali_test.csv \
  --model_id yali_10_30 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 10 \
  --label_len 5 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1
