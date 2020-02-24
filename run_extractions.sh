#!/usr/bin/env bash
WORDS=${1?Error: input a list of '-' seperated words}

python ./extract_features_json.py \
  --input_file=./example_sentences.json \
  --output_file=./features/$WORDS.jsonl \
  --vocab_file=./bert-base-uncased/vocab.txt \
  --bert_config_file=./bert-base-uncased/bert_config.json \
  --init_checkpoint=./bert-base-uncased/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8 \
  --words=$WORDS
