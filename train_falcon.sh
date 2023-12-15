python peft_train.py \
--model_name tiiuae/falcon-7b \
--max_seq_len 2048 \
--bf16 \
--group_by_length \
--bnb_4bit_compute_dtype bfloat16 \
--max_steps 200