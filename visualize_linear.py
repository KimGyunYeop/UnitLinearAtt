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
from utils import parse_args, load_bert_tokenizer
from argparse import Namespace

from tokenizer import SentencePeiceTokenizer
from models import AttentionModel
import json


args = parse_args()   
capture_range = args.capture_range

if args.result_path == "all":
    result_paths = os.listdir("results")
else:
    result_paths = [args.result_path]
    
exist_path = os.listdir("visualize")
    
for result_path in result_paths:
    
    if result_path == "README.md":
        continue
    
    result_path = os.path.join("results",result_path)
    file_list = os.listdir(result_path)

    try:
        with open(os.path.join(result_path,"config.json")) as f:
            json_object = json.load(f)
            print(json_object)

        with open(os.path.join(result_path,"result.json")) as f:
            result_dict = json.load(f)  
    except:
        continue
    
    args = vars(args)
    args.update(json_object)
    args = Namespace(**args)

    if args.no_attention or args.no_QKproj:
        continue

    if not os.path.isdir(os.path.join(result_path, str(args.epoch-1))):
        continue

    try:
        mt_type = "-".join([args.src_lang, args.tgt_lang])
        data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")
    except:
        mt_type = "-".join([args.tgt_lang, args.src_lang])
        data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")

    src_tokenizer, tgt_tokenizer = load_bert_tokenizer(args, data, mt_type)

    if capture_range is not None:
        args.result_path = args.result_path+"_cpt{}".format(str(capture_range))
    
    try:
        os.mkdir('./visualize/'+args.result_path)
    except:
        continue
        
        
    for i in range(12): # except 2 json file
        model = AttentionModel(args=args, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        model = model.to(args.gpu)
        model.load_state_dict(torch.load(os.path.join(result_path,str(i)+"/model_state_dict.pt"), map_location=f'cuda:{args.gpu}')) 
        model.eval()
        
        if args.softmax_linear:
            if args.act_type == "softmax":
                weight_act = nn.Softmax(dim=-1)
            elif args.act_type == "relu":
                weight_act = nn.ReLU()
            elif args.act_type == "leakyrelu":
                weight_act = nn.LeakyReLU(negative_slope=1.0/1000)
            else:
                assert "acy type error"
        
        if args.share_eye:
            if args.softmax_linear:
                try:
                    weight_numpy = weight_act(model.eye_linear.weights).cpu().detach().numpy()
                except:
                    weight_numpy = weight_act(model.eye_linear.weight).cpu().detach().numpy()
            else:
                weight_numpy = model.eye_linear.weight.cpu().detach().numpy()
            
            if capture_range is not None:
                weight_numpy=weight_numpy[:capture_range, :capture_range]
            plt.imshow(weight_numpy, 'hot')
            plt.colorbar()
            plt.savefig('./visualize/'+args.result_path+'/'+str(i)+'.png', dpi=1000)
            plt.clf()
        else:
            # print(repr(model.eye_linear_enc.weight))
            # print(repr(model.eye_linear_dec.weight))
            if args.softmax_linear:
                try:
                    enc_weight_numpy = weight_act(model.eye_linear_enc.weights).cpu().detach().numpy()
                except:
                    enc_weight_numpy = weight_act(model.eye_linear_enc.weight).cpu().detach().numpy()
                    
            else:
                enc_weight_numpy = model.eye_linear_enc.weight.cpu().detach().numpy()
                
            if args.softmax_linear:
                try:
                    dec_weight_numpy = weight_act(model.eye_linear_dec.weights).cpu().detach().numpy()
                except:
                    dec_weight_numpy = weight_act(model.eye_linear_dec.weight).cpu().detach().numpy()
            else:
                dec_weight_numpy = model.eye_linear_dec.weight.cpu().detach().numpy()
            
            
            if capture_range is not None:
                enc_weight_numpy=enc_weight_numpy[:capture_range, :capture_range]
                dec_weight_numpy=dec_weight_numpy[:capture_range, :capture_range]
                
                
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