python ../../../peft_train.py \
--model_name ../../../pretrain_models/bert-base-cased \
--max_seq_len 2048 \
--group_by_length \
--max_steps 200 \
--dataset_name ../../../text-classification/auditor_sentiment \
--num_labels 3 \
--epochs 5 \
--learning_rate 1e-3 \
--model_type SEQ_CLS \
--output_model_path ./result/bert-base-cased-auditor_sentiment-lora \
--bnb_4bit_compute_dtype float32 \
--use_4b False