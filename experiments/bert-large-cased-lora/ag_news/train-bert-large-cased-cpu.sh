python ../../../peft_train.py \
--model_name ../../../pretrain_models/bert-large-cased \
--max_seq_len 2048 \
--group_by_length \
--max_steps 200 \
--dataset_name ../../../text-classification/ag_news \
--num_labels 2 \
--epochs 5 \
--learning_rate 1e-3 \
--model_type SEQ_CLS \
--output_model_path ./result/bert-large-cased-ag_news-lora \
--bnb_4bit_compute_dtype float32 \
--use_4b False