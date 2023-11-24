#/bin/bash

result_path="en_de_lstm_base_reverse_dropout_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 1 \
    --no_attention \
    --source_reverse \
    --tokenizer_maxvocab 5000

    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 1

# python -u -m  train_attention.py \
#     --tokenizer_uncased \
#     --result_path "en_de_base_dot_1eye_softmax_linear7_test" \
#     --src_lang en \
#     --tgt_lang de \
#     --gpu 1 \
#     --softmax_linear \
#     --share_eye \
#     --alpha 7