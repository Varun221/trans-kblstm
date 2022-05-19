
from tqdm import tqdm
from tqdm import tqdm
import torch
import re
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def test(model, dataloader, TRAIN_MODEL):
    model.eval()
    correct = 0
    total = 0 
    gold_inds = []
    predictions_inds = []
    
    for batch_ndx, data in enumerate(tqdm(dataloader)):
        
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        prem_ind = data["prem_ind"].to(device, dtype=torch.long)
        hyp_ind = data["hyp_ind"].to(device, dtype=torch.long)
        kb_att = data["kb_att"].to(device, dtype=torch.long)
        prem_kb = data["prem_kb"].to(device, dtype=torch.float)
        # hyp_kb = data["hyp_kb"].to(device, dtype=torch.float)
        
        gold = data["labels"].to(device, dtype=torch.long)
        
        with torch.no_grad():

            if TRAIN_MODEL == 'trans_lstm':
                predictions = model(input_ids, attention_mask, prem_ind, hyp_ind, kb_att, prem_kb)
            else:
                predictions = model(input_ids, attention_mask)
            
        _ , inds = torch.max(predictions,1)
        gold_inds.extend(gold.tolist())
        predictions_inds.extend(inds.tolist())
        correct+= inds.eq(gold.view_as(inds)).cpu().sum().item()
        total += len(input_ids)

    model.train()
    return correct/total, gold_inds, predictions_inds




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


# read glove embeddings
def get_glove(dim):
    gloves = {}
    with open(f"../../data/glove.6B.{dim}d.txt", 'rb') as f:
        for l in tqdm(f, total=400000):
            line = l.decode().split()
            word = line[0]

            vect = np.array(line[1:]).astype(float)
            gloves[word] = vect

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False