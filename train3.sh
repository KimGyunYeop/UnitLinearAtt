#/bin/bash

result_path="en_de_dot_softmax_linear_2eye_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 3 \
    --softmax_linear \
    --source_reverse \
    --tokenizer_maxvocab 5000

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 3


