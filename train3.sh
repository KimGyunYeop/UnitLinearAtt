#/bin/bash

result_path="en_de_dot_softmax_linear_2eye_random_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 3 \
    --source_reverse \
    --tokenizer_maxvocab 5000 \
    --random_init

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 3


result_path="en_de_dot_softmax_linear_1eye_random_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 2 \
    --source_reverse \
    --tokenizer_maxvocab 5000 \
    --share_eye \
    --random_init

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 2
