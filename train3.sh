#/bin/bash

# result_path="en_de_dot_tie_5000"

# python -u -m  train_attention.py \
#     --result_path $result_path \
#     --model_type seq2seq\
#     --src_lang en \
#     --tgt_lang de \
#     --gpu 3 \
#     --source_reverse \
#     --tokenizer_maxvocab 5000 \
#     --no_QKproj \
#     --weight_tie

    
# python -u -m  test_bleu.py \
#     --result_path $result_path \
#     --gpu 3


result_path="en_de_lstm_tie_adam_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 3 \
    --source_reverse \
    --tokenizer_maxvocab 5000 \
    --no_attention \
    --weight_tie
    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 3
