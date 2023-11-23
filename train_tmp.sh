#/bin/bash

python -u -m  train_attention.py \
    --result_path "test" \
    --model_type "seq2seq" \
    --src_lang en \
    --tgt_lang de \
    --gpu 0 \
    --random_init 

