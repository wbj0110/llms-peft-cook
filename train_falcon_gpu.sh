python peft_train.py \
--model_name ybelkada/falcon-7b-sharded-bf16 \
--max_seq_len 2048 \
--bf16 \
--group_by_length \
--bnb_4bit_compute_dtype bfloat16 \
--max_steps 200