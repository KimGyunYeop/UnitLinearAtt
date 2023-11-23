from torch import nn
from torch.nn import functional as F
import torch
import math


class CustomLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, alpha):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.alpha=alpha

        weights = (torch.eye(size_out, size_in)*2-1)*self.alpha
        # print(weights)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter

    def forward(self, x):
        w_times_x= torch.matmul(x, F.softmax(self.weights,dim=-1).t())
        return w_times_x  # w times x + b

class AttentionModel(nn.Module):
    def __init__(self, args, src_tokenizer, tgt_tokenizer) -> None:
        super(AttentionModel, self).__init__()
        self.args = args

        self.src_emb = nn.Embedding(num_embeddings=args.tokenizer_maxvocab, embedding_dim=1000)
        self.tgt_emb = nn.Embedding(num_embeddings=args.tokenizer_maxvocab, embedding_dim=1000)

        self.enc_lstm = nn.LSTM(1000,1000, num_layers=4, batch_first=True, dropout=0.2)
        self.dec_lstm = nn.LSTM(1000,1000, num_layers=4, batch_first=True, dropout=0.2)
        
        self.tanh=nn.Tanh()
        
        if not args.no_attention:
            
            if not self.args.no_QKproj:
                
                if args.softmax_linear:
                    if args.share_eye:
                        self.eye_linear = CustomLinearLayer(1000,1000,args.alpha)
                    else:
                        self.eye_linear_enc = CustomLinearLayer(1000,1000,args.alpha)
                        self.eye_linear_dec = CustomLinearLayer(1000,1000,args.alpha)
                    
                else:
                    if args.share_eye:
                        self.eye_linear = nn.Linear(1000,1000,bias=False)
                        if not args.random_init:
                            self.eye_linear.weight.data = torch.nn.Parameter(
                                torch.eye(1000)
                            )
                    else:
                        self.eye_linear_enc = nn.Linear(1000,1000,bias=False)
                        self.eye_linear_dec = nn.Linear(1000,1000,bias=False)
                        
                        if not args.random_init:
                            self.eye_linear_enc.weight.data = torch.nn.Parameter(
                                torch.eye(1000)
                            )
                            
                            self.eye_linear_dec.weight.data = torch.nn.Parameter(
                                torch.eye(1000)
                            )
                    
            self.attn_linear=nn.Linear(1000*2,1000)

        self.tgt_lm_head = nn.Linear(1000, args.tokenizer_maxvocab)
        
        self.eos_token_id = tgt_tokenizer.eos_token_id
        self.bos_token_id = tgt_tokenizer.bos_token_id

    def forward(self, src, tgt):
        enc_out, (src_h_n, src_c_n) = self.enc(src) 
        logits = self.dec(enc_out, (src_h_n, src_c_n), tgt)

        return logits

    def enc(self, src):
        return self.enc_lstm(self.src_emb(src)) 
    
    def dec(self, enc_out, enc_hidden, tgt):
        src_h_n, src_c_n = enc_hidden
        
        dec_out, _ = self.dec_lstm(self.tgt_emb(tgt), (src_h_n, src_c_n))
        
        if not self.args.no_attention:
            if not self.args.no_QKproj:
                if self.args.share_eye:
                    dec_out=self.eye_linear(dec_out)
                    enc_out=self.eye_linear(enc_out)
                else:
                    dec_out=self.eye_linear_dec(dec_out)
                    enc_out=self.eye_linear_enc(enc_out)
                            
            attn_score=torch.matmul(dec_out,enc_out.transpose(1,2))
            attn_score=F.softmax(attn_score,dim=-1)
            attn_value=torch.matmul(attn_score,enc_out)
            
            dec_out=torch.cat([attn_value,dec_out],dim=-1)
            
            dec_out=self.attn_linear(dec_out)
            dec_out=self.tanh(dec_out)

        logits = self.tgt_lm_head(dec_out)
        
        return logits 
    
    def generate(self, src, max_len=50):
        enc_out, (src_h_n, src_c_n) = self.enc(src) 
        
        check_eos = [False] * src.size()[0] #[False, False, False, False]
        
        test_input = torch.Tensor([self.bos_token_id] * src.size()[0]).type(torch.long).to(src.device).view([src.size()[0],1]) # torch.Size([4, 1])

        final_logit = None
        
        for _ in range(max_len):
            logits = self.dec(enc_out, (src_h_n, src_c_n), test_input)
            
            if final_logit is None:
                final_logit = logits
            else:
                final_logit = torch.cat([final_logit, logits[:,-1:,:]], dim=1)
            
            pred_token = torch.argmax(F.softmax(logits[:,-1,:], dim=-1),dim=-1).view([src.size()[0],1]) # torch.Size([4, 50000]) / # torch.Size([4, 1]) # 이렇게 view로 해줘도 되나? 
            
            test_input = torch.cat([test_input, pred_token], dim=1)
            
            eos_batch=(pred_token == self.eos_token_id).nonzero(as_tuple=True)[0].tolist()
            for i in eos_batch:
                check_eos[i]=True
                
            if check_eos == ([True] * src.size()[0]):
                break
        
        return test_input, final_logit