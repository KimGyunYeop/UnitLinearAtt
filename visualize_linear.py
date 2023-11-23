import datasets
import transformers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.nn import functional as F

import os
from ssd1.gyop.research.various_attention.arg import parse_args
from argparse import Namespace

from tokenizer import SentencePeiceTokenizer
from models import AttentionModel
import json


args = parse_args()    
result_path = os.path.join("results",args.result_path)

with open(os.path.join("results",args.result_path,"config.json")) as f:
    json_object = json.load(f)
args = vars(args)
args.update(json_object)
args = Namespace(**args)

file_list = os.listdir(result_path) 
print(result_path)
print(file_list)
print()

try:
    mt_type = "-".join([args.src_lang, args.tgt_lang])
    data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")
except:
    mt_type = "-".join([args.tgt_lang, args.src_lang])
    data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")

# src_tokenizer = SentencePeiceTokenizer(uncased=args.tokenizer_uncased)
src_tokenizer = SentencePeiceTokenizer(uncased=args.tokenizer_uncased, max_vocab=args.tokenizer_maxvocab)
if os.path.isfile("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.src_lang)):
    src_tokenizer.load_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.src_lang))
else:
    src_tokenizer.make_vocab(data["train"], args.src_lang)
    src_tokenizer.save_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.src_lang))
# print(src_tokenizer.n_word)

tgt_tokenizer = SentencePeiceTokenizer(uncased=args.tokenizer_uncased, max_vocab=args.tokenizer_maxvocab)
if os.path.isfile("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.tgt_lang)):
    tgt_tokenizer.load_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.tgt_lang))
else:
    tgt_tokenizer.make_vocab(data["train"], args.tgt_lang)
    tgt_tokenizer.save_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.tgt_lang))
# print(tgt_tokenizer.n_word)

if args.capture_range is not None:
    args.result_path = args.result_path+"_cpt{}".format(str(args.capture_range))
    
os.makedirs('./visualize/'+args.result_path, exist_ok=True)
    
for i in range(len(file_list)-2): # except 2 json file
    model = AttentionModel(args=args, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    model = model.to(args.gpu)
    model.load_state_dict(torch.load(os.path.join(result_path,str(i)+"/model_state_dict.pt"), map_location=f'cuda:{args.gpu}')) 
    model.eval()
    
    # print(model)
    # assert 0
    if args.share_eye:
        # print(repr(model.eye_linear.weight))
        if args.softmax_linear:
            weight_numpy = F.softmax(model.eye_linear.weights, dim=-1).cpu().detach().numpy()
        else:
            weight_numpy = model.eye_linear.weight.cpu().detach().numpy()
        
        if args.capture_range is not None:
            weight_numpy=weight_numpy[:args.capture_range, :args.capture_range]
        plt.imshow(weight_numpy, 'hot')
        plt.colorbar()
        plt.savefig('./visualize/'+args.result_path+'/'+str(i)+'.png', dpi=1000)
        plt.clf()
    else:
        # print(repr(model.eye_linear_enc.weight))
        # print(repr(model.eye_linear_dec.weight))
        if args.softmax_linear:
            enc_weight_numpy = F.softmax(model.eye_linear_enc.weights, dim=-1).cpu().detach().numpy()
        else:
            enc_weight_numpy = model.eye_linear_enc.weight.cpu().detach().numpy()
            
        if args.softmax_linear:
            dec_weight_numpy = F.softmax(model.eye_linear_dec.weights, dim=-1).cpu().detach().numpy()
        else:
            dec_weight_numpy = model.eye_linear_dec.weight.cpu().detach().numpy()
        
        
        if args.capture_range is not None:
            enc_weight_numpy=enc_weight_numpy[:args.capture_range, :args.capture_range]
            dec_weight_numpy=dec_weight_numpy[:args.capture_range, :args.capture_range]
            
            
        plt.imshow(enc_weight_numpy, 'hot')
        plt.colorbar()
        plt.savefig('./visualize/'+args.result_path+'/'+str(i)+'_enc.png', dpi=1000)
        plt.clf()
        
        plt.imshow(dec_weight_numpy, 'hot')
        plt.colorbar()
        plt.savefig('./visualize/'+args.result_path+'/'+str(i)+'_dec.png', dpi=1000)
        plt.clf()
        
    print('./visualize/'+args.result_path+'/'+str(i))
    
# /home/nlplab/ssd1/gyop/research/various_attention/results/en_de_base_dot_2eye/19/model_state_dict.pt