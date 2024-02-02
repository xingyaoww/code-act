USE_TF=0

deepspeed --master_port 29510 \
		./ds_train.py \
	--cache_dir /net/nfs/path/to/cache/ \
        --model_name_or_path google/flan-t5-large \
        --output_dir model_ckpts/flan_large_0413 \
        --do_train \
	--do_eval \
	--save_total_limit=100 \
        --train_file ../data_utils/data_v5/fast_system.train.jsonl \
	--validation_file ../data_utils/data_v5/fast_system.val.jsonl \
	--predict_with_generate 0 \
        --learning_rate 1e-4 \
	--adam_eps 1e-06 \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 16 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 32 \
	--metric_for_best_model eval_loss \
	--greater_is_better=False \
	--deepspeed zero_2_bf16.json \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 8 \
	--logging_steps 1 \
	--load_best_model_at_end=True \
	--save_strategy=steps \
	--evaluation_strategy=steps \
	--save_steps 100 \
	--eval_steps 100 \
	--seed 42 \
	--report_to wandb \
	--run_name flan_large_0413
