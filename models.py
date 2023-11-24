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
    
    

#############################################

#clone by https://github.com/ljm565/neural-machine-translator-transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# word embedding layer
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output



# positional encoding layer
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_dim, pos_encoding):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pos_encoding = pos_encoding

        self.pos = torch.arange(0, self.max_len)
        if self.pos_encoding:
            self.pe = torch.zeros(self.max_len, self.hidden_dim)
            for i in range(0, self.hidden_dim, 2):
                self.pe[:, i] = np.sin(self.pos/(10000**(i/self.hidden_dim)))
                self.pe[:, i+1] = np.cos(self.pos/(10000**(i/self.hidden_dim)))         
            self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)
        else:
            self.emb_layer = nn.Embedding(self.max_len, self.hidden_dim)


    def forward(self, x):
        if self.pos_encoding:
            return self.pe[:, :x.size(1)]
        return self.emb_layer(self.pos.unsqueeze(0).to(x.device))[:, :x.size(1)]
        



# mulithead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_head, bias, self_attn, causal):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.bias = bias
        self.self_attn = self_attn
        self.causal = causal
        self.head_dim = self.hidden_dim // self.num_head
        assert self.hidden_dim == self.num_head * self.head_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)


    def head_split(self, x):
        x = x.view(self.batch_size, -1, self.num_head, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x


    def scaled_dot_product(self, q, k, v, mask):
        attn_wts = torch.matmul(q, torch.transpose(k, 2, 3))/(self.head_dim ** 0.5)
        if not mask == None:
            attn_wts = attn_wts.masked_fill(mask==0, float('-inf'))
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_out = torch.matmul(attn_wts, v)
        return attn_wts, attn_out


    def reshaping(self, attn_out):
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(self.batch_size, -1, self.hidden_dim)
        return attn_out


    def forward(self, query, key, value, mask):
        if self.self_attn:
            assert (query == key).all() and (key==value).all()

        self.batch_size = query.size(0)
        q = self.head_split(self.q_proj(query))
        k = self.head_split(self.k_proj(key))
        v = self.head_split(self.v_proj(value))

        attn_wts, attn_out = self.scaled_dot_product(q, k, v, mask)
        attn_out = self.attn_proj(self.reshaping(attn_out))

        return attn_wts, attn_out



# postion wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout, bias):
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.bias = bias

        self.FFN1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim, bias=self.bias),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(self.ffn_dim, self.hidden_dim, bias=self.bias),
        )
        self.init_weights()


    def init_weights(self):
        for _, param in self.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param.data, mean=0, std=0.02)

    
    def forward(self, x):
        output = self.FFN1(x)
        output = self.FFN2(output)
        return output



# a single encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.self_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=False)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.ffn_dim, self.dropout, self.bias)


    def forward(self, x, mask):
        attn_wts, output = self.self_attention(query=x, key=x, value=x, mask=mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        output = self.positionWiseFeedForward(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        return attn_wts, output



# all encoders
class Encoder(nn.Module):
    def __init__(self, config, tokenizer):
        super(Encoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id

        self.enc_num_layers = config.enc_num_layers
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.num_head = config.num_head
        self.max_len = config.max_len
        self.bias = bool(config.bias)
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps
        self.pos_encoding = config.pos_encoding
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEncoding(self.max_len, self.hidden_dim, self.pos_encoding)
        self.encoders = nn.ModuleList([EncoderLayer(self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.enc_num_layers)])


    def forward(self, x, mask=None):
        output = self.emb_layer(x) + self.pos_layer(x)
        output = self.dropout_layer(output)

        all_attn_wts = []
        for encoder in self.encoders:
            attn_wts, output = encoder(output, mask)
            all_attn_wts.append(attn_wts.detach().cpu())
        
        return all_attn_wts, output



# a single decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(DecoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.masked_self_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=True)
        self.enc_dec_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=False, causal=False)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.ffn_dim, self.dropout, self.bias)


    def forward(self, x, enc_output, dec_causal_mask, enc_dec_mask):
        dec_self_attn_wts, output = self.masked_self_attention(query=x, key=x, value=x, mask=dec_causal_mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        cross_attn_wts, output = self.enc_dec_attention(query=x, key=enc_output, value=enc_output, mask=enc_dec_mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        output = self.positionWiseFeedForward(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        return dec_self_attn_wts, cross_attn_wts, output



# all decoders
class Decoder(nn.Module):
    def __init__(self, config, tokenizer, ):
        super(Decoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id

        self.dec_num_layers = config.dec_num_layers
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.num_head = config.num_head
        self.max_len = config.max_len
        self.bias = bool(config.bias)
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps
        self.pos_encoding = config.pos_encoding

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEncoding(self.max_len, self.hidden_dim, self.pos_encoding)
        self.decoders = nn.ModuleList([DecoderLayer(self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.dec_num_layers)])


    def forward(self, x, enc_output, dec_causal_mask=None, enc_dec_mask=None):
        output = self.emb_layer(x) + self.pos_layer(x)
        output = self.dropout_layer(output)

        all_self_attn_wts, all_cross_attn_wts = [], []
        for decoder in self.decoders:
            dec_self_attn_wts, cross_attn_wts, output = decoder(output, enc_output, dec_causal_mask, enc_dec_mask)
            all_self_attn_wts.append(dec_self_attn_wts.detach().cpu())
            all_cross_attn_wts.append(cross_attn_wts.detach().cpu())
        
        return all_cross_attn_wts, output



# transformer
class Transformer(nn.Module):
    def __init__(self, config, src_tokenizer, tgt_tokenizer):
        super(Transformer, self).__init__()
        self.config = config
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = tgt_tokenizer
        
        self.hidden_dim = self.config.hidden_dim

        self.encoder = Encoder(self.config, self.src_tokenizer)
        self.decoder = Decoder(self.config, self.trg_tokenizer)
        self.fc = nn.Linear(self.hidden_dim, self.trg_tokenizer.vocab_size)
        
        self.tgt_bos_token_id = self.trg_tokenizer.bos_token_id


    def make_mask(self, src, trg):
        enc_mask = torch.where(src==self.src_tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        dec_causal_mask = torch.tril(torch.ones(trg.size(1), trg.size(1))).unsqueeze(0).unsqueeze(1).to(src.device) + torch.where(trg==self.trg_tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        dec_causal_mask = torch.where(dec_causal_mask < 2, 0, 1)
        enc_dec_mask = enc_mask
        return enc_mask, dec_causal_mask, enc_dec_mask


    def forward(self, src, trg):
        enc_mask, dec_causal_mask, enc_dec_mask = self.make_mask(src, trg)
        all_attn_wts, enc_output = self.encoder(src, enc_mask)
        all_cross_attn_wts, output = self.decoder(trg, enc_output, dec_causal_mask, enc_dec_mask)
        output = self.fc(output)
        return output
    
    def generate(self, src, trg):
        decoder_all_output = []
        decoder_all_logit = []
        for j in range(self.config.max_len):
            if j == 0:
                trg = trg[:, j].unsqueeze(1)
                _, output = self.model(src, trg)
                trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
            else:
                _, output = self.model(src, trg)
                trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
            decoder_all_logit.append(output[:, -1].unsqueeze(1))
            
        decoder_all_logit = torch.cat(decoder_all_logit, dim=1)
        decoder_all_output = torch.argmax(decoder_all_logit, dim=-1)
        
        return decoder_all_output, decoder_all_logit