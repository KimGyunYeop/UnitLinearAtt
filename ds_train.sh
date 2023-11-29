#/bin/bash

result_path="en_de_lstm_base_reverse_dropout_ds_test"

deepspeed --num_gpus=4 train_attention.py \
    --deepspeed \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --no_attention \
    --source_reverse

python -u -m  test.py \
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