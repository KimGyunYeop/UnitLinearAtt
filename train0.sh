#/bin/bash

result_path="en_de_transformer_base_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type transformer\
    --src_lang en \
    --tgt_lang de \
    --gpu 0 \
    --no_QKproj \
    --source_reverse \
    --tokenizer_maxvocab 5000

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 0


