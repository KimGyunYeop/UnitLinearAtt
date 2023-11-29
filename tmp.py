import torch
from torch.nn import functional as F

import datasets
import transformers

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import wandb

import os
import json
from tqdm import tqdm

from utils import parse_args, load_bert_tokenizer, load_word_tokenizer
from tokenizer import SentencePeiceTokenizer
from models import AttentionModel
from dataset import MTDataset
from nltk.translate.bleu_score import sentence_bleu
from torcheval.metrics import Perplexity
import evaluate
import subprocess

from argparse import Namespace


# for a in range(10):
#     print(a)
#     z=torch.zeros(1,1000)
#     z=z-a
#     # print(z)
#     # print(z.shape)

#     z[0,0]=a

#     # print(z)

#     z=F.softmax(z)

#     print(z[0,:2])
#     print(torch.sum(z))


# import datasets


# data = datasets.load_dataset("wmt14", "de-en", cache_dir="../../dataset/WMT")

# datalist = []
# for i in data["test"]:
#     datalist.append(i["translation"]["en"]+"\n")
    
# with open("tmp.txt","w") as fp:
#     for i in datalist:
#         fp.write(i)

# import transformers
# from transformers import BertLMHeadModel, T5ForConditionalGeneration

# model = T5ForConditionalGeneration.from_pretrained("t5-base")

# print(model)
# a = model.decoder.embed_tokens.weight
# print(a)
# b = model.lm_head.weight
# print(b)

# print(torch.max(a - b))



args = parse_args()    
device = "cuda:{}".format(str(args.gpu))
result_path = os.path.join("results",args.result_path)

with open(os.path.join("results",args.result_path,"config.json")) as f:
    json_object = json.load(f)
    
with open(os.path.join("results",args.result_path,"result.json")) as f:
    result_dict = json.load(f)    

args = vars(args)
args.update(json_object)
args = Namespace(**args)

# print(result_dict)
max_sacrebleu = 0
best_epoch = 0
eval_acc = 0
for e, i in result_dict.items():
    if i["sacrebleu"] >= max_sacrebleu:
        max_sacrebleu = i["sacrebleu"]
        eval_acc = i["accuracy"]
        best_epoch = e
print("base epoch=",best_epoch,"\t max eval sacrebleu=",max_sacrebleu, "\t eval acc=",eval_acc)

try:
    mt_type = "-".join([args.src_lang, args.tgt_lang])
    data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")
except:
    mt_type = "-".join([args.tgt_lang, args.src_lang])
    data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")

src_tokenizer, tgt_tokenizer = load_bert_tokenizer(args, data, mt_type)

if args.tokenizer_uncased:
    vocab_path = args.dataset+"_"+mt_type+"_"+"uncased_"+str(args.tokenizer_maxvocab)
else:
    vocab_path = args.dataset+"_"+mt_type+"_"+"cased_"+str(args.tokenizer_maxvocab)

try:
    data = datasets.load_from_disk(os.path.join("vocabs", vocab_path, "{}_{}_tokenized".format(args.src_lang, args.tgt_lang)))
except:
    assert "load vocab error"

data["test"] = data["test"].filter(lambda x:len(x["src_input_ids"]) < args.max_vocab and len(x["tgt_input_ids"]) < args.max_vocab)
test_dataset = MTDataset(data["test"], src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_key=args.src_lang, tgt_key=args.tgt_lang, source_reverse=args.source_reverse)

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

model = AttentionModel(args=args, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
model.load_state_dict(torch.load(os.path.join(result_path,str(best_epoch),'model_state_dict.pt'))) 

print(args)
print(model)
a = model.tgt_emb.weight
print(a)
b = model.tgt_lm_head.weight
print(b)

print(torch.max(a - b))