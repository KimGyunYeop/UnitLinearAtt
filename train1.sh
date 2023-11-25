#/bin/bash

result_path="en_de_dot_linear_1eye_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 1 \
    --share_eye \
    --source_reverse \
    --tokenizer_maxvocab 5000

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 1
    
result_path="en_de_dot_linear_1eye_50000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 1 \
    --share_eye \
    --source_reverse \
    --tokenizer_maxvocab 50000

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 1


result_path="en_de_dot_linear_1eye_random_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 1 \
    --share_eye \
    --source_reverse \
    --tokenizer_maxvocab 5000 \
    --random_init

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 1
