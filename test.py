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

def txt_write(path, data):
    with open(path, 'w') as f:
        f.writelines(data)
        
def cal_multi_bleu_perl(base_path, ref, pred):
    r = [s+'\n' for s in ref]
    p = [s+'\n' for s in pred]

    txt_write(base_path+'etc/ref.txt', r)
    txt_write(base_path+'etc/pred.txt', p)

    cmd = base_path+'etc/multi_bleu.perl ' + base_path+'etc/ref.txt < ' + base_path+'etc/pred.txt'
    result = subprocess.check_output(cmd, shell=True)

    os.remove(base_path+'etc/ref.txt')
    os.remove(base_path+'etc/pred.txt')
    
    return str(result)

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
model.cuda(device)

tgt_vocab_size = args.tokenizer_maxvocab
tgt_pad_token_id = test_dataset.tgt_pad_token_id

model.eval()

with torch.no_grad():
    num_correct = 0
    num_word = 0
    tmp = 0
    for batches, label, texts in tqdm(test_dataloader):
        
        batches = [batch.cuda(device) for batch in batches]
        label = label.cuda(device)

        logits = model(batches[0], batches[1])

        pred = torch.argmax(logits, dim=-1)

        correct_ids = label == pred
        label_mask = torch.where(torch.eq(label,-100), 0.0, 1.0)

        correct_ids = label_mask * correct_ids

        tmp += torch.sum(torch.eq(label, tgt_tokenizer.unk_token_id))
        num_correct += torch.sum(correct_ids)
        num_word += torch.sum(label_mask)

    accuracy = num_correct / num_word

    print("test_accurcay = ",accuracy.item())
    
    sacrebleu = evaluate.load("sacrebleu")
    
    bleu_list = []
    sacrebleu_list = []
    text_list = []
    pred_list = []
    ppl = Perplexity(ignore_index=-100)
    for batches, label, texts in tqdm(test_dataloader):
        batches = [batch.cuda(device) for batch in batches]
        label = label.cuda(device)

        outputs, test_logit = model.generate(batches[0], max_len=args.max_vocab)
        # print(outputs)
        
        batch_size, seq_len, prob_dim = test_logit.size()
        
        for i in range(batch_size):
            if args.tokenizer_uncased:
                ref = texts[1][i].lower()
            else:
                ref = texts[1][i]
                
            hyp = tgt_tokenizer.decode(outputs[i,:].tolist())
            # print(ref)
            # print(hyp)
            text_list.append(ref)
            pred_list.append(hyp)
            bleu_score = sentence_bleu([ref.split()], hyp.split())
            bleu_list.append(bleu_score)
            sacrebleu_score = sacrebleu.compute(predictions=[hyp], references=[ref])
            sacrebleu_list.append(sacrebleu_score["score"])
            
            # print(ref)
            # print(ref.split())
            # print(hyp)
            # print(hyp.split())
            # print(bleu_score)
            # print(sacrebleu_score)
            # print()
            
            # print(tgt_tokenizer.decode(batches[1][i,:]))
            # print(tgt_tokenizer.decode(label[i,:]))
        # ppl.update(test_logit, label)
        # print(ppl.compute())
    
    final_sacrebleu_score = sacrebleu.compute(predictions=pred_list, references=text_list)
            
    print("final BLEU = ",sum(bleu_list)/len(bleu_list))
    print("sacreBLEU = ",sum(sacrebleu_list)/len(sacrebleu_list))
    print("final sacreBLEU = ",final_sacrebleu_score["score"])
    print("\n\n")
    
    multi_bleu = cal_multi_bleu_perl("./", text_list, pred_list)
    print(multi_bleu)
    multi_bleu_score = float(multi_bleu.split(",")[0].split(" ")[-1])
    print("multi bleu = ",multi_bleu_score)
        