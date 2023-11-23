from torch.utils.data import Dataset
import torch


class MTDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer=None, tgt=True, src_key=None, tgt_key=None, source_reverse=False) -> None:
        super(MTDataset).__init__()
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        self.src_pad_token_id = src_tokenizer.pad_token_id
        if self.tgt_tokenizer is not None:
            self.tgt_pad_token_id = tgt_tokenizer.pad_token_id
        
        self.tgt=tgt
        self.source_reverse = source_reverse
        
        self.data = data
        
        if src_key is not None:
            self.src_key = src_key
        else:
            self.src_key = list(self.data[0]["translation"].keys())[0]
            
        if tgt_key is not None:
            self.tgt_key = tgt_key
        else:
            for i in list(self.data[0]["translation"].keys()):
                if self.src_key == i:
                    continue
                
                self.tgt_key = i
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]["translation"]
        
        src_text = text[self.src_key]
        src_token = self.data[index]["src_input_ids"]
        
        if self.tgt:
            tgt_text = text[self.tgt_key]
            tgt_token = self.data[index]["tgt_input_ids"]
            
            return {"src_token":src_token, "tgt_token":tgt_token, "src_text":src_text, "tgt_text":tgt_text}
        
        return {"src_token":src_token, "src_text":src_text}
    
    def collate_fn(self, batches):
        src_max_len = max([len(batch['src_token']) for batch in batches])
        
        if self.source_reverse:
            src_input_ids = [[self.src_pad_token_id] * (src_max_len - len(i['src_token'])) + i["src_token"][::-1] for i in batches]
        else:
            src_input_ids = [[self.src_pad_token_id] * (src_max_len - len(i['src_token'])) + i["src_token"] for i in batches]
            
        src_texts = [i["src_text"] for i in batches]
        
        src_input_ids = torch.LongTensor(src_input_ids)
        
        if self.tgt:
            tgt_max_len = max([len(batch['tgt_token']) for batch in batches])
            
            tgt_input_ids = [i["tgt_token"] + [self.tgt_pad_token_id] * (tgt_max_len - len(i['tgt_token'])) for i in batches]
            labels = [i["tgt_token"][1:] + [-100] * (tgt_max_len - len(i['tgt_token']) + 1) for i in batches]
            tgt_texts = [i["tgt_text"] for i in batches]
            
            tgt_input_ids = torch.LongTensor(tgt_input_ids)
            labels = torch.LongTensor(labels)
            
            return (src_input_ids, tgt_input_ids), labels, (src_texts, tgt_texts)
        
        return (src_input_ids), _, (src_texts)