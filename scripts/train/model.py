import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F



MODEL_NAME = 'roberta-large-mnli'

dims = {'roberta-base': [768, 384],
       'roberta-large': [1024, 512],
       'roberta-large-mnli': [1024, 512],
        'textattack/roberta-base-MNLI': [768, 384]}

def build_emb_matrix(vocab, word_dict, glove_dim):
    emb_mat = np.zeros((len(vocab)+1, glove_dim))
    for w, i in vocab.items():
        if w in word_dict:
            emb_mat[i] = word_dict[w]

    return emb_mat
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

LSTM_UNITS = 128
class Infotab_model(nn.Module):
    def __init__(self,model_name, tokenizer, vocab, word_dict, glove_dim, addtokenkb):

        super(Infotab_model, self).__init__()
        
        # lstm
        emb_mat = build_emb_matrix(vocab, word_dict, glove_dim)
        embed_size = emb_mat.shape[1]
        
        self.embedding = nn.Embedding(emb_mat.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_mat, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = SpatialDropout(0.3)

        # attention
        self.Prem_focus = nn.MultiheadAttention(2*LSTM_UNITS, 4, batch_first=True)
        self.reduce_prem = nn.Linear(768, 2*LSTM_UNITS)

        self.Hyp_focus = nn.MultiheadAttention(2*LSTM_UNITS, 4, batch_first=True)
        self.reduce_hyp = nn.Linear(768, 2*LSTM_UNITS)


        
        self.Prem_Encoder = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, 
                                    batch_first=True,
                                   num_layers = 2)
        
        self.Hyp_Encoder = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, 
                            batch_first=True,
                           num_layers = 2)
        
        
        # transformer
        self.model = RobertaModel.from_pretrained(MODEL_NAME)

        if addtokenkb:
            self.model.resize_token_embeddings(len(tokenizer))
        
        self.Lstm_red = nn.Sequential(
            nn.Linear(1792, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
    
        # for roberta-large
        self.fc1 = nn.Linear(2048, 512)
        self.drop_fin = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024+512, 512)
        self.final = nn.Linear(512, 3)
    

    
    def forward(self, input_ids, attention_mask, prem_ind, hyp_ind, kb_att, prem_kb):

        hyp_kb = torch.transpose(prem_kb, 1, 2)
        Transformer_out = self.model(input_ids, attention_mask = attention_mask)[1]
        
        # make attention masks
        prem_mask = (torch.logical_not(prem_ind == 0))
        hyp_mask = (torch.logical_not(hyp_ind == 0))
        
        prem_embed = self.embedding_dropout(self.embedding(prem_ind))
        hyp_embed = self.embedding_dropout(self.embedding(hyp_ind))
        
        # embedding shape - (batch-size, max length, embed_size) eg 2,200, 50
        
        prem_encoded, _ = self.Prem_Encoder(prem_embed)
        hyp_encoded, _ = self.Hyp_Encoder(hyp_embed)
        
        prem_encoded = prem_encoded*prem_mask.unsqueeze(2)
        hyp_encoded = hyp_encoded*hyp_mask.unsqueeze(2)

        # attention
        prem_kb_red = self.reduce_prem(torch.mean(prem_kb, dim=2))
        hyp_kb_red = self.reduce_hyp(torch.mean(hyp_kb, dim=2))

        prem_comp, prem_foc_wts = self.Prem_focus(hyp_encoded, prem_encoded, prem_kb_red)
        hyp_comp, hyp_foc_wts = self.Hyp_focus(prem_encoded, hyp_encoded, hyp_kb_red) 
        
        prem_know1 = (prem_foc_wts.unsqueeze(3) * prem_kb).sum(dim=2)
        hyp_know1 = (hyp_foc_wts.unsqueeze(3) * hyp_kb).sum(dim=2)

        prem_know2 = prem_encoded - prem_comp
        hyp_know2 = hyp_encoded - hyp_comp

        prem_know3 = prem_encoded * prem_comp
        hyp_know3 = hyp_encoded * hyp_comp
        
    
        prem_concat = torch.cat([prem_encoded, prem_comp, prem_know1, prem_know2, prem_know3], dim=2)
        hyp_concat = torch.cat([hyp_encoded, hyp_comp, hyp_know1, hyp_know2, hyp_know3], dim=2)
  
        prem_fin = self.Lstm_red(prem_concat)
        hyp_fin = self.Lstm_red(hyp_concat)
        # get poolings
        p_mean = torch.mean(prem_fin, dim=1)
        p_max = torch.max(prem_fin, dim=1).values
        h_mean = torch.mean(hyp_fin, dim=1)
        h_max = torch.max(hyp_fin, dim=1).values
        # all the above should be of size 1, 768
        f =  torch.cat([p_mean, p_max, h_mean, h_max], dim=-1)
        comb = torch.cat([f, Transformer_out], dim=-1)
        comb1 = self.fc1(comb)
        skipped = torch.cat([comb1, Transformer_out], dim=-1)
        reduced = self.fc2(self.drop_fin(skipped))
        out = self.final(F.relu(reduced))
        return comb1

class Infotab_Transonly_model(nn.Module):
    def __init__(self,model_name):
        super(Infotab_Transonly_model, self).__init__()
        self.model = RobertaModel.from_pretrained(MODEL_NAME)
        
        self.fc1 = nn.Linear(dims[model_name][0],dims[model_name][1])
        self.drop = torch.nn.Dropout(0.2)
        self.fc2 = nn.Linear(dims[model_name][1],3)
        #self.soft_max = torch.nn.Softmax(dim=1)
    def forward(self, enc, attention_mask):
        outputs = self.model(enc, attention_mask = attention_mask)
  
        out_intermediate = F.relu(self.fc1(outputs[1]))
        output = self.fc2(out_intermediate)

        return output