#/bin/bash

# result_path="en_de_2eye_tie_5000"

# python -u -m  train_attention.py \
#     --result_path $result_path \
#     --model_type seq2seq\
#     --src_lang en \
#     --tgt_lang de \
#     --gpu 2 \
#     --source_reverse \
#     --tokenizer_maxvocab 5000 \
#     --weight_tie

    
# python -u -m  test_bleu.py \
#     --result_path $result_path \
#     --gpu 2


# result_path="en_de_2eye_tie_softmax_5000"

# python -u -m  train_attention.py \
#     --result_path $result_path \
#     --model_type seq2seq\
#     --src_lang en \
#     --tgt_lang de \
#     --gpu 2 \
#     --source_reverse \
#     --tokenizer_maxvocab 5000 \
#     --weight_tie \
#     --softmax_linear

    
# python -u -m  test_bleu.py \
#     --result_path $result_path \
#     --gpu 2


result_path="en_de_lstm_adam_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 2 \
    --source_reverse \
    --no_attention \
    --tokenizer_maxvocab 5000
    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 2