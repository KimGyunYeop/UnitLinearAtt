import datasets
import transformers

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import wandb

import os
import json
from tqdm import tqdm

from utils import parse_args, load_word_tokenizer, load_bert_tokenizer
from tokenizer import SentencePeiceTokenizer
from models import AttentionModel, Transformer
from dataset import MTDataset
import evaluate
from argparse import Namespace

import deepspeed
from deepspeed.comm import comm

torch.manual_seed("1234")
torch.cuda.manual_seed_all("1234")

def make_dir(args):
    result_path = os.path.join("results",args.result_path)
    if "test" in args.result_path:
        os.makedirs(result_path, exist_ok=True)
    else:
        os.mkdir(result_path)

# mt_type = "de-en"
args = parse_args()
result_path = os.path.join("results",args.result_path)
device = "cuda:{}".format(str(args.gpu))

if args.deepspeed:
    comm.init_distributed("nccl")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    torch.cuda.set_device(args.local_rank)
    device = torch.cuda.current_device()
    
    if rank == 0:
        make_dir(args)

else:
    make_dir(args)


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
    vocab_path = args.dataset+"_"+mt_type+"_"+"uncased_"+str(args.tokenizer_maxvocab)
else:
    vocab_path = args.dataset+"_"+mt_type+"_"+"cased_"+str(args.tokenizer_maxvocab)

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
        # print(src_input_ids)
        tgt_input_ids = tgt_tokenizer.encode(examples["translation"][args.tgt_lang])
        
        return {"src_input_ids":src_input_ids, "tgt_input_ids":tgt_input_ids}

    data = data.map(group_texts, num_proc=1, load_from_cache_file=False)
    os.makedirs(os.path.join("vocabs", vocab_path, "{}_{}_tokenized".format(args.src_lang, args.tgt_lang)),exist_ok=True)
    data.save_to_disk(os.path.join("vocabs", vocab_path, "{}_{}_tokenized".format(args.src_lang, args.tgt_lang)))

print(data)
print("filtering...")
# data = data.filter(lambda x:len(x["translation"][args.src_lang].split()) <= args.max_word and len(x["translation"][args.src_lang].split()) <= args.max_word)
data = data.filter(lambda x:len(x["src_input_ids"]) <= args.max_vocab and len(x["tgt_input_ids"]) <= args.max_vocab, load_from_cache_file=False)
print(data)

train_dataset = MTDataset(data["train"], src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_key=args.src_lang, tgt_key=args.tgt_lang, source_reverse=args.source_reverse)
dev_dataset = MTDataset(data["validation"], src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_key=args.src_lang, tgt_key=args.tgt_lang, source_reverse=args.source_reverse)
test_dataset = MTDataset(data["test"], src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_key=args.src_lang, tgt_key=args.tgt_lang, source_reverse=args.source_reverse)

print("deepspeed",args.deepspeed)
if args.deepspeed:
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                                  sampler=DistributedSampler(train_dataset, num_replicas=world_size), num_workers=4)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn,
                                sampler=DistributedSampler(train_dataset, num_replicas=world_size), num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn,
                                 sampler=DistributedSampler(train_dataset, num_replicas=world_size), num_workers=4)
    
else:
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=4)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn, num_workers=4)


# for batches, label, texts in tqdm(train_dataloader):
#     batches = [batch.cuda(device) for batch in batches]
#     print(texts[0][0:5])
#     print(texts[1][0:5])
#     print(batches[0].size())
#     print(src_tokenizer.tokenizer.batch_decode(batches[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)[0:5])
#     print(batches[1].size())
#     print(tgt_tokenizer.tokenizer.batch_decode(batches[1],skip_special_tokens=True, clean_up_tokenization_spaces=True)[0:5])
#     print(tgt_tokenizer.tokenizer.batch_decode(label,skip_special_tokens=True, clean_up_tokenization_spaces=True)[0:5])
#     print(label)
#     print(tgt_tokenizer.unk_token_id)
#     assert 0

if args.model_type == "seq2seq":
    model = AttentionModel(args=args, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    
elif args.model_type == "transformer":
    transformer_config = json.load(open("transformer_config.json", "r"))
    transformer_config["vocab_size"] = args.tokenizer_maxvocab
    args = vars(args)
    args.update(transformer_config)
    args = Namespace(**args)
    
    model = Transformer(config=args, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    
else:
    assert "error model type"

if args.deepspeed:
    if args.model_type == "seq2seq":
        args.deepspeed_config = "ds_config_seq2seq.json"
        
    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters())
    
else:
    if args.model_type == "seq2seq":
        model.cuda(device)
        # optimizer = SGD(model.parameters(), lr=1)
        # scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.5)
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=4, gamma=0.8)
    elif args.model_type == "transformer":
        model.cuda(device)
        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9,0.98), eps=0.0001)
        scheduler = StepLR(optimizer, step_size=4, gamma=0.8)
    else:
        assert "error model type"

print(args)
print(model)
# optimizer = Adam(model.parameters(), lr=args.learning_rate)
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
with open(os.path.join(result_path, "model.txt"), "w") as text_file:
    text_file.write(str(model))
result_dict = {}
step = 0
for e in range(args.epoch):
    if args.deepspeed:
        engine.train()
    else:
        model.train()
    
    for name, child in model.named_children():
        for param in child.parameters():
            if 'eye_linear' in name:           
                param.requires_grad = True if e>=args.warmup_epochs else False
                print(name)
                print(param)
                print(param.requires_grad)
                    
    for batches, label, texts in tqdm(train_dataloader):
        batches = [batch.cuda(device) for batch in batches]
        label = label.cuda(device)

        if args.deepspeed:
            logits = engine.forward(batches[0], batches[1])
            loss = lf(logits.view(-1, tgt_vocab_size), label.view(-1))
            
            engine.backward(loss)
            engine.step()
            
        else:
            logits = model(batches[0], batches[1])
            loss = lf(logits.view(-1, tgt_vocab_size), label.view(-1))
        
            if args.model_type == "seq2seq":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
            elif args.model_type == "transformers":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                
                

        step += 1

    if args.deepspeed:
        engine.eval()
    else:
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


            if args.deepspeed:
                logits = engine.forward(batches[0], batches[1])
            else:
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
    if args.deepspeed:
        engine.save_checkpoint(os.path.join(save_path,"ds"))
        
        json.dump(result_dict, open(os.path.join(result_path, "result_{}.json".format(rank)), "w"), indent=2)
        
    torch.save(model.state_dict(), os.path.join(save_path,'model_state_dict.pt'))
    torch.save(optimizer.state_dict(), os.path.join(save_path,'optimizer.pt'))
    json.dump(result_dict, open(os.path.join(result_path, "result.json"), "w"), indent=2)
    
    if args.model_type == "seq2seq":
        scheduler.step()
        # if e >= args.warmup_schedule:
        #     for param_group in optimizer.param_groups: 
        #         param_group['lr'] = param_group['lr']*0.5
    elif args.model_type == "transformer":
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