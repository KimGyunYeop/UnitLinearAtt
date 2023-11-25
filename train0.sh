#/bin/bash

result_path="en_de_dot_reverse_dropout_50000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 0 \
    --no_QKproj \
    --source_reverse \
    --tokenizer_maxvocab 50000

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 0


result_path="en_de_lstm_base_reverse_dropout_50000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 0 \
    --no_attention \
    --source_reverse \
    --tokenizer_maxvocab 50000

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 0


result_path="en_de_linear_2eye_random_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 2 \
    --source_reverse \
    --tokenizer_maxvocab 50000 \
    --random_init

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 2