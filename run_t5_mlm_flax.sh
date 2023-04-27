python run_t5_mlm_flax.py \
--output_dir="./outputs/vi-flan-t5-base" \
--model_type="t5" \
--model_name_or_path="google/flan-t5-base" \
--tokenizer_name="./outputs/vi-flan-t5-base" \
--train_file="./data/vie_wikipedia_2021_1M/vie_wikipedia_2021_1M-sentences.txt" \
--validation_file="./data/vie_wikipedia_2021_300K/vie_wikipedia_2021_300K-sentences.txt" \
--max_seq_length="512" \
--per_device_train_batch_size="8" \
--per_device_eval_batch_size="8" \
--learning_rate="0.005" \
--weight_decay="0.001" \
--warmup_steps="2000" \
--overwrite_output_dir \
--adafactor \
--do_train \
--do_eval \
--logging_steps="500" \
--save_steps="1000" \
--eval_steps="1000"