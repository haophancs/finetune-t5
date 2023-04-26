python run_t5_mlm_flax.py \
	--output_dir="./outputs" \
	--model_type="t5" \
	--model_name_or_oath="google/flan-t5-base" \
	--tokenizer_name="./outputs" \
  --train_file="./data/vie_wikipedia_2021_1M/vie_wikipedia_2021_1M-sentences.txt" \
  --validation_file="./data/vie_wikipedia_2021_300K/vie_wikipedia_2021_300K-sentences.txt" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500"