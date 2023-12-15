python ../../../peft_train.py \
--model_name ../../../pretrain_models/flan-t5-base \
--max_seq_len 2048 \
--group_by_length \
--max_steps 200 \
--dataset_name ../../../text-classification/financial_phrasebank \
--num_labels 3 \
--epochs 5 \
--learning_rate 1e-3\
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--model_type SEQ_2_SEQ_LM \
--output_model_path ./result/flan-t5-base-financial-lora \
--bnb_4bit_compute_dtype float16 \
--need_hyperparameters_search False \
--enable_peft False \
--use_4b False
