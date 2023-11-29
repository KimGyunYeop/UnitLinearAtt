#/bin/bash

# result_path="en_de_dot_softmax_linear_2eye_tie_5000"

# python -u -m  train_attention.py \
#     --result_path $result_path \
#     --model_type seq2seq\
#     --src_lang en \
#     --tgt_lang de \
#     --gpu 0 \
#     --source_reverse \
#     --tokenizer_maxvocab 5000 \
#     --weight_tie \
#     --softmax_linear

    
# python -u -m  test_bleu.py \
#     --result_path $result_path \
#     --gpu 0


# result_path="en_de_dot_softmax_linear_2eye_tie_warmup5_5000"

# python -u -m  train_attention.py \
#     --result_path $result_path \
#     --model_type seq2seq\
#     --src_lang en \
#     --tgt_lang de \
#     --gpu 0 \
#     --source_reverse \
#     --tokenizer_maxvocab 5000 \
#     --weight_tie \
#     --softmax_linear \
#     --warmup_epochs 5

    
# python -u -m  test_bleu.py \
#     --result_path $result_path \
#     --gpu 0


result_path="en_de_dot_adam_5000"

python -u -m  train_attention.py \
    --result_path $result_path \
    --model_type seq2seq\
    --src_lang en \
    --tgt_lang de \
    --gpu 0 \
    --source_reverse \
    --no_QKproj \
    --tokenizer_maxvocab 5000
    
python -u -m  test_bleu.py \
    --result_path $result_path \
    --gpu 0