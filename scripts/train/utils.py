
from tqdm import tqdm


from tqdm import tqdm
import torch






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