from torch import nn
from torch.nn import functional as F
import torch
import math


class CustomLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, args, size_in, size_out):
        super().__init__()
        self.args = args
        self.size_in, self.size_out = size_in, size_out
        weights = torch.eye(size_out, size_in)
        self.alpha=args.alpha
        if args.act_type == "softmax":
            self.weight_act = nn.Softmax(dim=-1)
            self.alpha=args.alpha
            weights = (weights*2-1)*self.alpha
        elif args.act_type == "relu":
            self.weight_act = nn.ReLU()
            self.alpha=args.alpha
        elif args.act_type == "leakyrelu":
            self.weight_act = nn.LeakyReLU(negative_slope=1.0/size_in)
        else:
            assert "acy type error"

        
        # print(weights)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter

    def forward(self, x):
        w_times_x= torch.matmul(x, self.weight_act(self.weights).t())
        return w_times_x  # w times x + b
    
class AttentionModel(nn.Module):
    def __init__(self, args, src_tokenizer, tgt_tokenizer) -> None:
        super(AttentionModel, self).__init__()
        self.args = args

        self.src_emb = nn.Embedding(num_embeddings=args.tokenizer_maxvocab, embedding_dim=1000, padding_idx=src_tokenizer.pad_token_id)
        self.tgt_emb = nn.Embedding(num_embeddings=args.tokenizer_maxvocab, embedding_dim=1000, padding_idx=tgt_tokenizer.pad_token_id)

        self.enc_lstm = nn.LSTM(1000,1000, num_layers=4, batch_first=True, dropout=0.2)
        self.dec_lstm = nn.LSTM(1000,1000, num_layers=4, batch_first=True, dropout=0.2)
        
        self.tanh=nn.Tanh()
        
        if not args.no_attention:
            
            if not self.args.no_QKproj:
                
                if args.softmax_linear:
                    if args.share_eye:
                        self.eye_linear = CustomLinearLayer(self.args, 1000,1000)
                    else:
                        self.eye_linear_enc = CustomLinearLayer(self.args, 1000,1000)
                        self.eye_linear_dec = CustomLinearLayer(self.args, 1000,1000)
                    
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
        self.pad_token_id = tgt_tokenizer.pad_token_id
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer
        
        if self.args.weight_tie:
            self.tgt_lm_head.weight = self.tgt_emb.weight
            
            if getattr(self.tgt_lm_head, "bias", None) is not None:
                self.tgt_lm_head.bias.data = nn.functional.pad(
                    self.tgt_lm_head.bias.data,
                    (
                        0,
                        self.tgt_lm_head.weight.shape[0] - self.tgt_lm_head.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            

    def forward(self, src, tgt):
        enc_out, (src_h_n, src_c_n) = self.enc(src) 
        logits = self.dec(enc_out, (src_h_n, src_c_n), tgt, src)

        return logits

    def enc(self, src):
        return self.enc_lstm(self.src_emb(src)) 
    
    def dec(self, enc_out, enc_hidden, tgt, src):
        src_h_n, src_c_n = enc_hidden
        
        enc_dec_mask = torch.where(src==self.src_tokenizer.pad_token_id, 0, 1).unsqueeze(1)
        
        dec_out, _ = self.dec_lstm(self.tgt_emb(tgt), (src_h_n, src_c_n))
        
        if not self.args.no_attention:
            if not self.args.no_QKproj:
                if self.args.share_eye:
                    dec_out=self.eye_linear(dec_out)
                    enc_out=self.eye_linear(enc_out)
                else:
                    dec_out=self.eye_linear_dec(dec_out)
                    enc_out=self.eye_linear_enc(enc_out)
                            
            attn_score=torch.matmul(dec_out,enc_out.transpose(1,2)).masked_fill(enc_dec_mask==0, float('-inf'))
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
            logits = self.dec(enc_out, (src_h_n, src_c_n), test_input, src)
            
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
    
    
    def generate_beam_search(self, src, max_len=50, beam_size=5, a=0.6, min_length=5):
        enc_out, (src_h_n, src_c_n) = self.enc(src) 
        
        batch_size = src.size()[0]
        beam_size_puls_one = beam_size+1
        
        check_eos = np.array([[False] * beam_size] * batch_size)
        batch_clear = np.array([False] * batch_size)
        
        test_first_input = torch.Tensor([self.bos_token_id] * batch_size).type(torch.long).to(src.device).view([batch_size,1]) # torch.Size([4, 1])
        test_input = torch.Tensor([[self.bos_token_id] * beam_size] * batch_size).type(torch.long).to(src.device).view([batch_size, beam_size, 1])
        # test_cumul_prob = torch.ones([batch_size, beam_size], dtype=torch.float64).to(src.device)
        test_cumul_prob = torch.zeros([batch_size, beam_size], dtype=torch.float64).to(src.device)
        result_tensor = torch.ones([batch_size, beam_size, max_len]).type(torch.long).to(src.device) * self.pad_token_id
        result_prob = torch.zeros([batch_size, beam_size], dtype=torch.float64).to(src.device)
        
        logits = self.dec(enc_out, (src_h_n, src_c_n), test_first_input, src)
        # word_prob = F.softmax(logits[:,-1,:], dim=-1)
        word_prob = torch.log(F.softmax(logits[:,-1,:], dim=-1))
        pred_token = torch.topk(word_prob, k=beam_size, dim=-1)
        
        test_input = torch.cat([test_input, pred_token.indices.unsqueeze(-1)],dim=-1)
        # test_cumul_prob = test_cumul_prob * pred_token.values
        test_cumul_prob = test_cumul_prob + pred_token.values
        
        
        for i in range(1, max_len-1):
            
            test_cumul_prob_tmp = test_cumul_prob.unsqueeze(-1).expand(-1, -1, beam_size_puls_one)
            
            cand_index = []
            cand_prob = []
            
            for bms in range(beam_size):
                logits = self.dec(enc_out, (src_h_n, src_c_n), test_input[:, bms, :], src)
                # word_prob = F.softmax(logits[:,-1,:], dim=-1)
                word_prob = torch.log(F.softmax(logits[:,-1,:], dim=-1))
                pred_token = torch.topk(word_prob, k=beam_size_puls_one, dim=-1)
                
                # beam_cumul_prob = test_cumul_prob_tmp[:, bms, :] * pred_token.values
                beam_cumul_prob = test_cumul_prob_tmp[:, bms, :] + pred_token.values
                cand_prob.append(beam_cumul_prob.unsqueeze(1))
                
                cumul_index = torch.cat([test_input[:, bms, :].unsqueeze(1).expand(-1,beam_size_puls_one,-1), pred_token.indices.unsqueeze(-1)], dim=-1)
                cand_index.append(cumul_index.unsqueeze(1))
                
            cand_index = torch.cat(cand_index, dim=1)
            cand_prob = torch.cat(cand_prob, dim=1)
            
            while True:
                select_topk = torch.topk(cand_prob.view(batch_size, -1), k=beam_size, dim=-1)
                
                #get cumul probabiltiy of beam size
                test_cumul_prob = select_topk.values
                
                #get cumul index(seqeuence) of beam size
                cand_index = cand_index.view(batch_size, beam_size * beam_size_puls_one, i+2)
                test_input = []
                for bs_index in range(batch_size):
                    test_input.append(cand_index[bs_index,...].index_select(dim=0, index=select_topk.indices[bs_index, :]).unsqueeze(0))
                
                test_input = torch.cat(test_input, dim=0)
                
                eos_index = test_input[..., -1] == self.eos_token_id
                eos_index = eos_index * torch.Tensor(~batch_clear).unsqueeze(-1).to(eos_index.device)
                
                if torch.sum(eos_index) == 0:
                    
                    # tmp = test_input.view(-1,i+2)
                    # for t in range(beam_size*2):
                    #     print(self.tgt_tokenizer.decode(tmp[t,:].tolist()))
                    # print("\n\n")
                    
                    break
                else:
                    result_cand_index = eos_index.nonzero()
                    for j in range(result_cand_index.size()[0]):
                        top_index = result_cand_index[j,:].tolist()
                        
                        tmp_result = test_input[*top_index,:]
                        
                        if batch_clear[top_index[0]]:
                            continue
                        
                        result_tensor[top_index[0], sum(check_eos[top_index[0]]), :tmp_result.size()[-1]] = tmp_result
                        # print(top_index[0], sum(check_eos[top_index[0]]), tmp_result.size()[-1])
                        # print(self.tgt_tokenizer.decode(tmp_result.tolist()))
                        # print(result_tensor[top_index[0], sum(check_eos[top_index[0]])])
                        # print(tmp_result)
                        # print(result_tensor[*top_index,:])
                        # print(result_tensor[52, 0, :])
                        
                        tmp_prob = test_cumul_prob[*top_index]
                        # print(tmp_prob)
                        #add penalty
                        tmp_prob = tmp_prob * (((1.0 + (i+2)) ** a) / ((1.0 + min_length) ** a))
                        result_prob[top_index[0], sum(check_eos[top_index[0]])] = tmp_prob
                        # print(test_cumul_prob[result_cand_index[j,0],:])
                        # print(result_prob[top_index[0], sum(check_eos[top_index[0]])])
                        # print("-----------------\n\n")
                        
                        check_eos[top_index[0], sum(check_eos[top_index[0]])] = True
                        # print(check_eos)
                        
                        if sum(check_eos[top_index[0]]) == beam_size:
                            batch_clear[top_index[0]] = True
                        
                        cand_prob.view(batch_size, -1)[top_index[0], select_topk.indices[*top_index]] = -float('inf')
            
            # print(test_input.shape)
            # print(sum(batch_clear))
            if sum(batch_clear) == batch_size:
                break
        
        if not sum(batch_clear) == batch_size:
            for j in range(batch_size):
                if not batch_clear[j]:
                    for e, k in enumerate(range(sum(check_eos[j]), beam_size)):
                        if check_eos[j, k]:
                            assert 0
                            
                        tmp_result = test_input[j, e,:]
                        result_tensor[j, k, :max_len] = tmp_result[:max_len]
                        result_tensor[j,k,-1] = self.eos_token_id
                        
                        tmp_prob = test_cumul_prob[j,e]
                        tmp_prob = tmp_prob * (((1.0 + (i+2)) ** a) / ((1.0 + min_length) ** a))
                        result_prob[j,k] = tmp_prob
        
        # print("\n\n")
        # print(result_prob)
        top1_cand = torch.argmax(result_prob, dim=-1)
        # print(top1_cand)
        # print(result_tensor.shape)
        # result_tensor = result_tensor.index_select(dim=1, index=top1_cand)
        
        final_result = []
        for i in range(batch_size):
            final_result.append(result_tensor[i:i+1,top1_cand[i]])
        
        final_result = torch.cat(final_result, dim=0)
        
        return final_result, None
    
    

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
    def __init__(self, args, hidden_dim, num_head, bias, self_attn, causal):
        super(MultiHeadAttention, self).__init__()
        self.args = args
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
        
        if not self.args.no_QKproj:
            if args.softmax_linear:
                if args.share_eye and self.self_attn:
                    self.eye_linear = CustomLinearLayer(self.args, self.head_dim, self.head_dim)
                else:
                    self.eye_linear_q = CustomLinearLayer(self.args, self.head_dim,self.head_dim)
                    self.eye_linear_k = CustomLinearLayer(self.args, self.head_dim,self.head_dim)
                
            else:
                if args.share_eye and self.self_attn:
                    self.eye_linear = nn.Linear(self.head_dim,self.head_dim,bias=False)
                    
                    if not args.random_init:
                        self.eye_linear.weight.data = torch.nn.Parameter(
                            torch.eye(self.head_dim)
                        )
                else:
                    self.eye_linear_q = nn.Linear(self.head_dim,self.head_dim,bias=False)
                    self.eye_linear_k = nn.Linear(self.head_dim,self.head_dim,bias=False)
                    
                    if not args.random_init:
                        self.eye_linear_q.weight.data = torch.nn.Parameter(
                            torch.eye(self.head_dim)
                        )
                        
                        self.eye_linear_k.weight.data = torch.nn.Parameter(
                            torch.eye(self.head_dim)
                        )


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
        
        if not self.args.no_QKproj:
            if self.args.share_eye:
                q = self.eye_linear(q)
                k = self.eye_linear(k)
            else:
                q = self.eye_linear_q(q)
                k = self.eye_linear_k(k)

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
    def __init__(self, args, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.self_attention = MultiHeadAttention(self.args, self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=False)
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
        
        self.args = config
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEncoding(self.max_len, self.hidden_dim, self.pos_encoding)
        self.encoders = nn.ModuleList([EncoderLayer(self.args, self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.enc_num_layers)])


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
    def __init__(self, args, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(DecoderLayer, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.masked_self_attention = MultiHeadAttention(self.args, self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=True)
        self.enc_dec_attention = MultiHeadAttention(self.args, self.hidden_dim, self.num_head, self.bias, self_attn=False, causal=False)
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
        
        self.args = config

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEncoding(self.max_len, self.hidden_dim, self.pos_encoding)
        self.decoders = nn.ModuleList([DecoderLayer(self.args, self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.dec_num_layers)])


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
        
        if self.config.weight_tie:
            self.fc.weight = self.decoder.emb_layer.emb_layer.weight
            
            if getattr(self.fc, "bias", None) is not None:
                self.fc.bias.data = nn.functional.pad(
                    self.fc.bias.data,
                    (
                        0,
                        self.fc.weight.shape[0] - self.fc.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
        
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