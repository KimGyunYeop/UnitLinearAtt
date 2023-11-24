from tqdm import tqdm
import json
import operator
import os

from transformers import BertTokenizer

class Tokenizer_wmt:
    def __init__(self, args, mt_type, lang=None):
        
        if args.tokenizer_uncased:
            vocab_path = args.dataset+"_"+mt_type+"_"+"uncased_"+str(args.tokenizer_maxvocab)
        else:
            vocab_path = args.dataset+"_"+mt_type+"_"+"cased_"+str(args.tokenizer_maxvocab)

        self.tokenizer = BertTokenizer(vocab_file=os.path.join("vocabs", vocab_path, lang, "vocab.txt"), do_lower_case=args.tokenizer_uncased) 

        
        self.special_tokens = {
            "bos_token":self.tokenizer.cls_token,
            "eos_token":self.tokenizer.sep_token,
            "pad_token":self.tokenizer.pad_token,
            "unk_token":self.tokenizer.unk_token
        }

        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.bos_token, self.bos_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def encode(self, s):
        return self.tokenizer.encode(s)

    def decode(self, tok, reverse=False):
        if reverse:
            tok = tok[::-1]

        try:
            tok = tok[:tok.index(self.eos_token_id)]
        except ValueError:
            try:
                tok = tok[:tok.index(self.pad_token_id)]
            except:
                pass
        
        try:
            if tok[0] == self.bos_token_id:
                tok = tok[1:]
        except IndexError:
            tok=[]

        return self.tokenizer.decode(tok)

class SentencePeiceTokenizer():
    def __init__(self, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>", uncased=False, max_vocab=None) -> None:
        # self.bos_token = bos_token
        # self.eos_token = eos_token
        # self.pad_token = pad_token
        # self.unk_token = unk_token
        
        self.special_tokens = {
            "bos_token":bos_token,
            "eos_token":eos_token,
            "pad_token":pad_token,
            "unk_token":unk_token
        }
        
        self.uncased = uncased
        self.max_vocab = max_vocab

        self.token2id = {}
        self.id2token = []
        self.count_word = {}

        self.n_word = 0
        
        for i in self.special_tokens.values():
            self.add_word(i)
        
        # self.add_word(self.bos_token)
        # self.add_word(self.eos_token)
        # self.add_word(self.pad_token)
        # self.add_word(self.unk_token)

    def make_vocab(self, dataset, key):
        for sent in tqdm(dataset, desc="make tokenzier..."):
            if self.uncased:
                sent["translation"][key] = sent["translation"][key].lower()
            for word in sent["translation"][key].split(" "):
                self.add_word(word)
                
        if self.max_vocab is not None and self.max_vocab < self.n_word:
            tmp_count_word = self.count_word.copy()
            
            for i in self.special_tokens.values():
                del(tmp_count_word[i])
            
            print("sorting by counting word...")
            sorted_count_word = sorted(tmp_count_word.items(), key=operator.itemgetter(1), reverse=True)
             
            print("cutting by max vocab...")
            sorted_count_word = dict(sorted_count_word[:self.max_vocab - len(list(self.special_tokens.values()))])
            
            self.id2token = list(self.special_tokens.values()) + list(sorted_count_word.keys())
            self.token2id = dict(zip(self.id2token, range(len(self.id2token))))
            
            self.n_word = len(self.id2token)           

    def add_word(self, word):
        if word not in self.token2id.keys():
            self.token2id[word] = self.n_word
            self.id2token.append(word)
            self.count_word[word] = 1

            self.n_word += 1
        else:
            self.count_word[word] += 1

    def encode(self, sent):
        input_ids = []
        input_ids.append(self.token2id[self.special_tokens["bos_token"]])
        
        if self.uncased:
            sent = sent.lower()
        
        for word in sent.split(" "):
            if word in self.token2id.keys():
                input_ids.append(self.token2id[word])
            else:
                input_ids.append(self.token2id[self.special_tokens["unk_token"]])

        input_ids.append(self.token2id[self.special_tokens["eos_token"]])

        return input_ids
    
    def decode(self, ids, skip_spc_token=True):
        sent = " ".join(list(map(lambda x : self.id2token[x], ids)))
        if skip_spc_token:
            sent = sent.split("</s>")[0]
            if sent[0:4] == "<s> ":
                sent = sent[4:]

        return sent
    
    def save_vocab(self, file):
        json.dump({"token2id":self.token2id, "id2token":self.id2token, "count_word":self.count_word, "n_word":self.n_word, "uncased":self.uncased, "special_tokens":self.special_tokens}, open(file, "w", encoding="utf8"))

    def load_vocab(self, file):
        tmp = json.load(open(file, "r", encoding="utf8"))

        self.token2id = tmp["token2id"]
        self.id2token = tmp["id2token"]
        self.count_word = tmp["count_word"]
        self.n_word = tmp["n_word"]
        self.uncased = tmp["uncased"]
        self.special_tokens = tmp["special_tokens"]