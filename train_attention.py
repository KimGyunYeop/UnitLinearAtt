import datasets
import transformers

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import wandb

import os
import json
from tqdm import tqdm

from utils import parse_args, load_word_tokenizer, load_bert_tokenizer
from tokenizer import SentencePeiceTokenizer
from models import AttentionModel
from dataset import MTDataset
import evaluate

# mt_type = "de-en"
args = parse_args()
device = "cuda:{}".format(str(args.gpu))
result_path = os.path.join("results",args.result_path)
if "test" in args.result_path:
    os.makedirs(result_path, exist_ok=True)
else:
    os.mkdir(result_path)

try:
    mt_type = "-".join([args.src_lang, args.tgt_lang])
    data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")
except:
    mt_type = "-".join([args.tgt_lang, args.src_lang])
    data = datasets.load_dataset(args.dataset, mt_type, cache_dir="../../dataset/WMT")

# src_tokenizer, tgt_tokenizer = load_word_tokenizer(args, data, mt_type)
src_tokenizer, tgt_tokenizer = load_bert_tokenizer(args, data, mt_type)

# print(data["train"][0]["translation"][args.src_lang])
# a = src_tokenizer.encode(data["train"][0]["translation"][args.src_lang])
# print(a)
# print(src_tokenizer.decode(a))
# print(data["train"][0]["translation"][args.tgt_lang])
# b = tgt_tokenizer.encode(data["train"][0]["translation"][args.tgt_lang])
# print(b)
# print(tgt_tokenizer.decode(b))

# assert 0

#pre process dataset


if args.tokenizer_uncased:
    vocab_path = args.dataset+"_"+mt_type+"_"+"uncased"
else:
    vocab_path = args.dataset+"_"+mt_type+"_"+"cased"

try:
    print("load dataset...")
    data = datasets.load_from_disk(os.path.join("vocabs", vocab_path, "{}_{}_tokenized".format(args.src_lang, args.tgt_lang)))
except:
    # import multiprocessing
    # num_proc = multiprocessing.cpu_count()
    print("pre processing dataset...")
    num_proc = 1
    
    def group_texts(examples):
        src_input_ids = src_tokenizer.encode(examples["translation"][args.src_lang])
        tgt_input_ids = tgt_tokenizer.encode(examples["translation"][args.tgt_lang])
        
        return {"src_input_ids":src_input_ids, "tgt_input_ids":tgt_input_ids}

    data = data.map(group_texts, num_proc=num_proc)
    os.makedirs(os.path.join("vocabs", vocab_path, "{}_{}_tokenized".format(args.src_lang, args.tgt_lang)),exist_ok=True)
    data.save_to_disk(os.path.join("vocabs", vocab_path, "{}_{}_tokenized".format(args.src_lang, args.tgt_lang)))

print(data)
print("filtering...")
data = data.filter(lambda x:len(x["src_input_ids"]) < 51 and len(x["tgt_input_ids"]) < 51)
print(data)
# data["train"] = data["train"][:500]
# print(data["train"])
train_dataset = MTDataset(data["train"], src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_key=args.src_lang, tgt_key=args.tgt_lang)
dev_dataset = MTDataset(data["validation"], src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_key=args.src_lang, tgt_key=args.tgt_lang)
test_dataset = MTDataset(data["test"], src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_key=args.src_lang, tgt_key=args.tgt_lang)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

if args.model_type == "seq2seq":
    model = AttentionModel(args=args, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    model.cuda(device)
    optimizer = SGD(model.parameters(), lr=1)
else:
    assert "error model type"

print(model)
# optimizer = Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.5)
lf = nn.CrossEntropyLoss()
    
sacrebleu = evaluate.load("sacrebleu")

tgt_vocab_size = args.tokenizer_maxvocab
tgt_pad_token_id = train_dataset.tgt_pad_token_id

# wandb.init(
#     project="Various Attention",
#     name=args.result_path,
#     config=args,
# )        

json.dump(vars(args), open(os.path.join(result_path, "config.json"), "w"), indent=2)
result_dict = {}
step = 0
for e in range(args.epoch):
    model.train()
    
    for name, child in model.named_children():
        for param in child.parameters():
            if 'eye_linear' in name:           
                param.requires_grad = True if e>=args.warmup_epochs else False
                print(name)
                print(param)
                print(param.requires_grad)
                    
    for batches, label, texts in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        batches = [batch.cuda(device) for batch in batches]
        label = label.cuda(device)

        logits = model(batches[0], batches[1])
        # print(logits.shape)
        loss = lf(logits.view(-1, tgt_vocab_size), label.view(-1))
        # print(loss)
        loss.backward()
        
        if args.model_type == "seq2seq":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
        optimizer.step()

        step += 1

    model.eval()
    text_list = []
    pred_list = []
    with torch.no_grad():
        num_correct = 0
        num_word = 0
        tmp = 0
        for batches, label, texts in tqdm(dev_dataloader):
            
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
            
            batch_size, seq_len, prob_dim = logits.size()
        
            for i in range(batch_size):
                if args.tokenizer_uncased:
                    ref = texts[1][i].lower()
                else:
                    ref = texts[1][i]
                hyp = tgt_tokenizer.decode(pred[i,:].tolist())
                text_list.append(ref)
                pred_list.append(hyp)
                
        
        final_sacrebleu_score = sacrebleu.compute(predictions=pred_list, references=text_list)

        accuracy = num_correct / num_word

        print(tmp/num_word)
        print(accuracy)
        result_dict[str(e)] = {"accuracy":accuracy.item(), "sacrebleu":final_sacrebleu_score["score"], "unk_percent":(tmp/num_word).item()}
    
    save_path = os.path.join(result_path, str(e))
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path,'model_state_dict.pt'))
    torch.save(optimizer.state_dict(), os.path.join(save_path,'optimizer.pt'))
    json.dump(result_dict, open(os.path.join(result_path, "result.json"), "w"), indent=2)
    
    if e >= args.warmup_schedule:
        scheduler.step()
    
######################################################################################################            
    
# print(tgt_tokenizer.n_word)
# print(tgt_tokenizer.id2token[:10])
# print(list(tgt_tokenizer.token2id.items())[:10])
# print(tgt_tokenizer.token2id["wiederaufnahme"])

# print(data["train"][0]["translation"]["en"])
# a = src_tokenizer.tokenize(data["train"][0]["translation"]["en"].lower())
# print(a)
# print(src_tokenizer.decode(a))

# print(data["train"][0]["translation"]["de"])
# a = tgt_tokenizer.tokenize(data["train"][0]["translation"]["de"].lower())
# print(a)
# print(tgt_tokenizer.decode(a))