#/bin/bash

result_path="en_de_transformer_2eye_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type transformer\
    --src_lang en \
    --tgt_lang de \
    --gpu 1 \
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