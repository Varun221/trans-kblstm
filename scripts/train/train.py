# # imports
import pandas as pd
import numpy as np
import csv
import os
import random
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from model import Infotab_model, Infotab_Transonly_model
from utils import test


import time
import copy
from tqdm.notebook import tqdm

from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse
parser = argparse.ArgumentParser(description='get arguments')

parser.add_argument("--addkb", type=bool, default=False, help="Select whether to add knowledge. default is False")
parser.add_argument("--addtokenkb", type=bool, default=False, help="Select whether to append wordnet tokens to Transformer inputs. default is False")
parser.add_argument("--makekbrandom", type=bool, default=False, help="Select whether to add knowledge as random noise")
parser.add_argument("--batchsize", type=int, default=3, help="select batch size")
parser.add_argument("--glove_dim", type=int, default=300, help="Glove dimension")
parser.add_argument("--seed", type=int, default=42, help="Seed for training")
parser.add_argument("--data_percent", type=int, default=50, help="Percentage of data to train")
parser.add_argument("--gold", type=bool, default=False, help="Training on Gold data")
parser.add_argument("--kg_exp", type=bool, default=False, help="Training on Knowledge et al")
parser.add_argument("--savepath", type=str, default="./", help="Model Save path")

args = parser.parse_args()


# get glove embeds
gloves = {}
with open(f"./glove.6B.{args.glove_dim}d.txt", 'rb') as f:
    for l in tqdm(f, total=400000):
        line = l.decode().split()
        word = line[0]

        vect = np.array(line[1:]).astype(float)
        gloves[word] = vect


# nltk for wordnet
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re
stop_words = list(set(stopwords.words('english')))
stop_words.extend(['(', ')'])
stop_words = set(stop_words)

# remove numbers and special characters to check words only
def clean_sentence(line):
    
    line = line.lower()
    
    line = line.strip()

    line = re.sub(r'\:(.*?)\:','',line)
    line = re.sub('\[.*?\]', '', line)
    line = re.sub('<.*?>+', '', line)
    line = re.sub('\n', '', line)
    
    for ch in ")(:-.,°′\"\'%$+;/\#&?!_":
        line = line.replace(ch, f" {ch} ")
    return line

# load wordnet
f = open("../../data/pair_features.txt")
lines = f.readlines()
kb_dict = {}
wordnet_words = set()
for line in tqdm(lines):
    w1 = line.split(' ')[0].split(';')[0]
    w2 = line.split(' ')[0].split(';')[1]
    wordnet_words.add(w1)
    wordnet_words.add(w2)
    vals = line.split(' ')[1:]
    vals[-1] = vals[-1].replace('\n', '')
    vals = np.array([float(x) for x in vals])
    if w1 in kb_dict:
        kb_dict[w1][w2] = vals
    else:
        kb_dict[w1] = {}
        kb_dict[w1][w2] = vals
f.close()


def Add_knowledge(prem, hyp):
    premwords = set(clean_sentence(prem).split())
    hypwords = set(clean_sentence(hyp).split())
    knwsent = " <KNW> "
    rel_pairs = set()
    for w1 in premwords:
        for w2 in hypwords:
            if (w1 in kb_dict):
                if (w2 in kb_dict[w1]):
                    if(w1!=w2):
                        if(w1 > w2):
                            rel_pairs.add((w2, w1, 1))
                        else:
                            rel_pairs.add((w1, w2, 0))
            elif (w2 in kb_dict):
                if (w1 in kb_dict[w2]):
                    if(w1!=w2):
                        if(w1 > w2):
                            rel_pairs.add((w2, w1, 0))
                        else:
                            rel_pairs.add((w1, w2, 1))
    knwsent = "<KNW> "
    for p in rel_pairs:
        knwsent += f" [ {p[0]} : {p[1]} ; "
        if(p[2] == 1):
            vec = kb_dict[p[0]][p[1]]
        else:
            vec = kb_dict[p[1]][p[0]]
        if(vec[0] != 0):
            knwsent += "<HYPE> "
        if(vec[1] != 0):
            knwsent += "<HYPO> "
        if(vec[2] != 0):
            knwsent += "<CO_HYP> "
        if(vec[3] != 0):
            knwsent += "<ANT> "
        if(vec[4] != 0):
            knwsent += "<SYN> "
        knwsent += "] "
    final = "<s> " + prem + " </s> " + hyp + " " + knwsent
    return final

# clone repositories for data
# !git clone https://github.com/utahnlp/knowledge_infotabs.git
# !git clone https://github.com/infotabs/infotabs.git

# Make Word collection
#@title Make word collection
print("Loading datasets...")
dfs = []

if args.gold:
    dfs.append(pd.read_csv("../../data/Gold_Lstm_bert/Gold_a1_withlstmrels.csv"))
    dfs.append(pd.read_csv("../../data/Gold_Lstm_bert/Gold_a2_withlstmrels.csv"))
    dfs.append(pd.read_csv("../../data/Gold_Lstm_bert/Gold_a3_withlstmrels.csv"))
    dfs.append(pd.read_csv("../../data/Gold_Lstm_bert/Gold_train_withlstmrels.csv"))
    dfs.append(pd.read_csv("../../data/Gold_Lstm_bert/Gold_dev_withlstmrels.csv"))
elif args.kg_exp:
    dfs.append(pd.read_csv("../../data/kg_explicit/train.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("../../data/kg_explicit/dev.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("../../data/kg_explicit/test_alpha1.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("../../data/kg_explicit/test_alpha2.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("../../data/kg_explicit/test_alpha3.tsv", delimiter='\t'))
else:
    dfs.append(pd.read_csv("./knowledge_infotabs/temp/data/drr_ablation/train.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("./knowledge_infotabs/temp/data/drr_ablation/dev.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("./knowledge_infotabs/temp/data/drr_ablation/test_alpha1.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("./knowledge_infotabs/temp/data/drr_ablation/test_alpha2.tsv", delimiter='\t'))
    dfs.append(pd.read_csv("./knowledge_infotabs/temp/data/drr_ablation/test_alpha3.tsv", delimiter='\t'))
    


sents = []
for df in dfs:
    sents.extend(df['premise'])
    sents.extend(df['hypothesis'])
    
print(f"Loaded {len(dfs)} datasets")
print("\nBuilding vocab...")
vocab = {}
# will store complete vocabulary in format : word : index
ind = 1
for sent in tqdm(sents):
    cln = clean_sentence(sent)
    
    for w in cln.split():
        if w not in vocab:
            vocab[w] = ind
            ind += 1  
print(f"Completed building vocab dictionary with {len(vocab)} words")
print(f"\nChecking the Glove embeddings...")
nf = 0
for w in vocab.keys():
    if w not in gloves:
        nf += 1
print(f"{nf*100/len(vocab) : .1f} % words not found in glove datasets")



# save memory
import gc
all_w = set()
print("Making word collection... ")
for sent in tqdm(sents):
    cln = clean_sentence(sent)
    for w in cln.split():
        all_w.add(w)
print("Word collection made\nDeleting excess words...")
word_dict = {}
for k, v in gloves.items():
    if k in all_w:
        word_dict[k] = v
del(gloves)
print("Extra words deleted")
gc.collect()


MODEL_NAME = "roberta-large-mnli"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

if args.addtokenkb:
    tokenizer.add_special_tokens({'additional_special_tokens': ["<KNW>", "<SYN>", "<ANT>", "<HYPE>", "<HYPO>", "<CO_HYP>"]})


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(args.seed)

if args.gold:
    train = pd.read_csv("../../data/Gold_Lstm_bert/Gold_train_withlstmrels.csv")
elif args.kg_exp:
    train = pd.read_csv("../../data/kg_explicit/train.tsv", delimiter='\t')    
else:
    train = pd.read_csv("../../data/Infotabs/temp/train_with_lstmrels.csv")

# debug
if args.data_percent == 100:
    pass
else:
    train = train.sample(frac=1.0, random_state=2022).reset_index(drop=True)
    train = train.loc[:int((args.data_percent/100.) * len(train))]


# RELATIONAL MAPPING
Rel_mapping = {
    'Antonym': 'is opposite of',
    'AtLocation': 'is at location',
    'CapableOf': 'is capable of',
    'Causes': 'causes',
    'CausesDesire': 'causes desire to',
    'CreatedBy': 'is created by',
    'DefinedAs': 'is defined as',
    'DerivedFrom': 'is derived from',
    'Desires': 'desires',
    'DistinctFrom': 'is distinct from',
    'Entails': 'entailes',
    'EtymologicallyDerivedFrom': 'is etymologically derived from',
    'EtymologicallyRelatedTo': 'is etymologically related to',
    'ExternalURL': 'external url',
    'FormOf': 'is a form of',
    'HasA': 'has a',
    'HasContext': 'has context',
    'HasFirstSubevent': 'has first subevent',
    'HasLastSubevent': 'has last subevent',
    'HasPrerequisite': 'has prerequisite',
    'HasProperty': 'has property',
    'HasSubevent': 'has subevent',
    'InstanceOf': 'is an instance of',
    'IsA': 'is a',
    'LocatedNear': 'is located near',
    'MadeOf': 'is made of',
    'MannerOf': 'is manner of',
    'MotivatedByGoal': 'is motivated by goal',
    'NotCapableOf': 'is not capable of',
    'NotDesires': 'does not desire',
    'NotHasProperty': 'does not have property',
    'PartOf': 'is part of',
    'ReceivesAction': 'receives action',
    'RelatedTo': 'is related to',
    'SimilarTo': 'is similar to',
    'SymbolOf': 'is a symbol of',
    'Synonym': 'is same as',
    'UsedFor': 'is used for',
    'dbpedia/capital': 'has capital',
    'dbpedia/field': 'has field',
    'dbpedia/genre': 'has genre',
    'dbpedia/genus': 'has genus',
    'dbpedia/influencedBy': 'is influenced by',
    'dbpedia/knownFor': 'is known for',
    'dbpedia/language': 'has language',
    'dbpedia/leader': 'has leader',
    'dbpedia/occupation': 'has occupation',
    'dbpedia/product': 'has product',
    'Hypernym': 'is hypernym of',
    'Hyponym': 'is hyponym of',
    'Co-Hyponym': 'is co-hyponym of'
    
}

Rels_list = list(Rel_mapping.keys())

senttrans = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# compute embeddings
Embeds = {}
for k, v in tqdm(Rel_mapping.items()):
    Embeds[k] = senttrans.encode([v], show_progress_bar=False)


def get_rel_vector(rels):
    Embeddings = [Embeds[x] for x in rels]
    finenc = np.mean(np.array(Embeddings), axis=0)
    return finenc



LSTM_MAX_LEN = 200
import ast

class Infotab_dataset(Dataset):
    
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        prem = self.df.loc[idx, 'premise']
        hyp = self.df.loc[idx, 'hypothesis']
        lab = self.df.loc[idx, 'label']
        fullinp = "<s> " + prem + " </s> " + hyp + " </s>"

        if args.addtokenkb:
            fullinp = Add_knowledge(prem, hyp)
        
        inp_tokens = self.tokenizer.encode_plus(fullinp, 
                                          padding="max_length", 
                                          add_special_tokens=False,
                                          max_length=512,
                                        truncation=True)
        # lstm knowledge addition
        prem_cln_words = clean_sentence(prem).split()
        hyp_cln_words = clean_sentence(hyp).split()
        prem_ind = [vocab[w] for w in prem_cln_words]
        hyp_ind = [vocab[w] for w in hyp_cln_words]
        # pad sequences
        if(len(prem_ind) > LSTM_MAX_LEN):
            prem_ind = prem_ind[:LSTM_MAX_LEN]
        else:
            pad_len = LSTM_MAX_LEN - len(prem_ind)
            prem_ind = prem_ind + [0,]*pad_len
            
        
        if(len(hyp_ind) > LSTM_MAX_LEN):
            hyp_ind = hyp_ind[:LSTM_MAX_LEN]
        else:
            pad_len = LSTM_MAX_LEN - len(hyp_ind)
            hyp_ind = hyp_ind + [0,]*pad_len
            
        if not args.makekbrandom:
            kb_att = np.zeros((LSTM_MAX_LEN, LSTM_MAX_LEN))
            prem_kb = np.zeros((LSTM_MAX_LEN, LSTM_MAX_LEN, 768))
            hyp_kb = np.zeros((LSTM_MAX_LEN, LSTM_MAX_LEN, 768))


            if not args.kg_exp:
                if self.df.loc[idx, 'Rel_with_toks'] == self.df.loc[idx, 'Rel_with_toks']:
                    rels = ast.literal_eval(self.df.loc[idx, 'Rel_with_toks'])

                    for rel in rels:
                        
                        if rel[2] == 0:
                            if rel[0][1] < LSTM_MAX_LEN and rel[1][1] < LSTM_MAX_LEN:
                                kb_att[rel[0][1], rel[1][1]] = 1.0
                                prem_kb[rel[0][1], rel[1][1], :] = get_rel_vector(rel[3])
                        else:
                            if rel[0][1] < LSTM_MAX_LEN and rel[1][1] < LSTM_MAX_LEN:
                                kb_att[rel[1][1], rel[0][1]] = 1.0
                                hyp_kb[rel[0][1], rel[1][1], :] = get_rel_vector(rel[3])
                    
        if args.makekbrandom:
            kb_att = np.random.rand(LSTM_MAX_LEN, LSTM_MAX_LEN)
            prem_kb = np.random.rand(LSTM_MAX_LEN, LSTM_MAX_LEN, 768)
            hyp_kb = np.random.rand(LSTM_MAX_LEN, LSTM_MAX_LEN, 768)
        
        return {
                "input_ids": torch.tensor(inp_tokens.input_ids, dtype=torch.long),
                "attention_mask":torch.tensor(inp_tokens.attention_mask, dtype=torch.long),
                "prem_ind": torch.tensor(prem_ind, dtype=torch.long),
                "hyp_ind": torch.tensor(hyp_ind, dtype=torch.long),
                "kb_att": torch.tensor(kb_att, dtype=torch.long),
                "prem_kb": torch.tensor(prem_kb, dtype=torch.float) if args.addkb else torch.tensor(0),
                # "hyp_kb": torch.tensor(hyp_kb, dtype=torch.float),
                "labels":torch.tensor(lab, dtype=torch.long)
            }
        



# MODEL
import pickle as pkl

if args.addkb:
    TRAIN_MODEL = 'trans_lstm'
else:
    TRAIN_MODEL = 'trans'


if TRAIN_MODEL == 'trans_lstm':
    model = Infotab_model(MODEL_NAME, tokenizer, vocab, word_dict, args.glove_dim, args.addtokenkb)
else:
    model = Infotab_Transonly_model(MODEL_NAME)

if TRAIN_MODEL == 'trans_lstm':
    all_params = dict(model.named_parameters())
    lstm_params = [p for k,p in all_params.items() if (('Hyp_Encoder' in k) or ('Prem_Encoder' in k))]
    other_params = [p for k,p in all_params.items() if not (('Hyp_Encoder' in k) or ('Prem_Encoder' in k))]

    print(len(lstm_params), len(other_params), len(all_params))


# moving the model and defining the optimizer
model = model.to(device)
criterion = nn.CrossEntropyLoss()

LEARNING_RATE_LSTM = 1e-3
LEARNING_RATE_TRANS = 1e-4

if TRAIN_MODEL == 'trans_lstm':
    optimizer = optim.Adagrad(
        [
            {"params": lstm_params, "lr": LEARNING_RATE_LSTM},
            {"params": other_params},
        ],
        lr = LEARNING_RATE_TRANS,
    )

else:
    optimizer = optim.Adagrad(list(model.parameters()), lr=LEARNING_RATE_TRANS)

PATIENCE = 5
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, )


if args.gold:
    alpha1 = pd.read_csv("../../data/Gold_Lstm_bert/Gold_a1_withlstmrels.csv")
    alpha2 = pd.read_csv("../../data/Gold_Lstm_bert/Gold_a2_withlstmrels.csv")
    alpha3 = pd.read_csv("../../data/Gold_Lstm_bert/Gold_a3_withlstmrels.csv")
    dev = pd.read_csv("../../data/Gold_Lstm_bert/Gold_dev_withlstmrels.csv")

elif args.kg_exp:
    alpha1 = pd.read_csv("../../data/kg_explicit/test_alpha1.tsv", delimiter='\t')
    alpha2 = pd.read_csv("../../data/kg_explicit/test_alpha2.tsv", delimiter='\t')
    alpha3 = pd.read_csv("../../data/kg_explicit/test_alpha3.tsv", delimiter='\t')
    dev = pd.read_csv("../../data/kg_explicit/dev.tsv", delimiter='\t')

else:
    alpha1 = pd.read_csv("../../data/Infotabs/temp/a1_with_lstmrels.csv")
    alpha2 = pd.read_csv("../../data/Infotabs/temp/a2_with_lstmrels.csv")
    alpha3 = pd.read_csv("../../data/Infotabs/temp/a3_with_lstmrels.csv")
    dev = pd.read_csv("../../data/Infotabs/temp/dev_with_lstmrels.csv")


# dataloaders
train_dataset = Infotab_dataset(train, tokenizer)
val_dataset = Infotab_dataset(dev, tokenizer)
a1_dt = Infotab_dataset(alpha1, tokenizer)
a2_dt = Infotab_dataset(alpha2, tokenizer)
a3_dt = Infotab_dataset(alpha3, tokenizer)

train_dataloader=DataLoader(train_dataset,batch_size=args.batchsize,num_workers=2,shuffle=True, pin_memory=True)
val_dataloader=DataLoader(val_dataset,batch_size=args.batchsize,num_workers = 2,shuffle=False, pin_memory=True)
a1_dataloader=DataLoader(a1_dt,batch_size=args.batchsize,num_workers=2,shuffle=False, pin_memory=True)
a2_dataloader=DataLoader(a2_dt,batch_size=args.batchsize,num_workers = 2,shuffle=False, pin_memory=True)
a3_dataloader=DataLoader(a3_dt,batch_size=args.batchsize,num_workers = 2,shuffle=False, pin_memory=True)


model.train()

print("Gold: ", args.gold, "  Percent: ", args.Data_percent, "  Seed: ", args.seed, "  Add_Knowledge: ", args.addkb)

noinc = 0 
best_acc = 0
best_dict = None

# start training
for ep in range(25):
    
    if noinc > PATIENCE:
        break

    epoch_loss = 0
    start = time.time()
    
    total = 0
    correct = 0
    gold_inds = []
    
    for batch_ndx, data in enumerate(tqdm(train_dataloader)):
        
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        prem_ind = data["prem_ind"].to(device, dtype=torch.long)
        hyp_ind = data["hyp_ind"].to(device, dtype=torch.long)
        
        kb_att = data["kb_att"].to(device, dtype=torch.long)
        prem_kb = data["prem_kb"].to(device, dtype=torch.float)        
        gold = data["labels"].to(device, dtype=torch.long)
        
        batch_loss = 0

        if TRAIN_MODEL == 'trans_lstm':
            predictions = model(input_ids, attention_mask, prem_ind, hyp_ind, kb_att, prem_kb)
        else:
            predictions = model(input_ids, attention_mask)
        
        _ , inds = torch.max(predictions,1)
        gold_inds.extend(gold.tolist())
        correct+= inds.eq(gold.view_as(inds)).cpu().sum().item()
        total += len(input_ids)
        out_loss = criterion(predictions,gold)
        out_loss.backward()
        batch_loss+=out_loss.item()
        epoch_loss+=batch_loss
        optimizer.step()
        optimizer.zero_grad()

    normalized_epoch_loss = epoch_loss/(len(train_dataloader))
    print("Epoch {}".format(ep+1))
    print("Epoch loss: {} ".format(normalized_epoch_loss))

    dev_acc, dev_gold, dev_pred = test(model ,val_dataloader, MODEL_NAME)
    print("Dev Accuracy: {}".format(dev_acc))

    scheduler.step(dev_acc)

    if dev_acc > best_acc:
        best_acc = dev_acc
        best_dict = model.state_dict()
        noinc = 0

        torch.save({'model_state_dict': model.state_dict(),
                    'epoch': ep+1, 
                    },
                 f"{args.save_path}")

    else:
        noinc += 1
    

model.load_state_dict(best_dict)

a1_acc, _, _ = test(model ,a1_dataloader, MODEL_NAME)
print(a1_acc)
a2_acc, _, _ = test(model ,a2_dataloader, MODEL_NAME)
print(a2_acc)
a3_acc, _, _ = test(model ,a3_dataloader, MODEL_NAME)
print(a3_acc)

end = time.time()
print("Train Accuracy: {}".format(correct/total))
print("Dev accuracy: {}".format(best_acc))
print("a1 Accuracy: {}".format(a1_acc))
print("a2 Accuracy: {}".format(a2_acc))
print("a3 Accuracy: {}".format(a3_acc))

print("Time taken: {} seconds\n".format(end-start))
print("\n\n")
