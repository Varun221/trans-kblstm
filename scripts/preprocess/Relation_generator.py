import pandas as pd
from tqdm import tqdm
import argparse

# using stopwords from nltk
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import re

# relation mappings stored in ../train/relation_templates.json
import json

parser = argparse.ArgumentParser(description='Get_file_name')

parser.add_argument('--path',  metavar='path',  type=str, help='the path to file')
parser.add_argument("--savename", metavar='savename', type=str, help="file name to save")

# Execute the parse_args() method
args = parser.parse_args()

file_path = args.path
save_path = args.savename

if file_path[-3:] == 'csv':
    df = pd.read_csv(file_path)
else:
    df = pd.read_csv(file_path, delimiter='\t')



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
    
    for ch in ")(:-.,°′`\"\'%$+;/\#&?!_":
        line = line.replace(ch, f" {ch} ")

    return line


# get knowledge database
comb = pd.read_csv("../../data/Conceptnet_Wordnet_full.csv")

with open("../train/relation_templates.json") as json_file:
    Rel_mapping = json.load(json_file)


Rels_list = list(Rel_mapping.keys())

# all possible unique wordpairs in the dataset
word_pairs = set(list(zip(comb['word1'], comb['word2'])))


'''
Each premise and hypothesis pair is appended with a list of relational connections. 

(word1, index of word1), (word2, index of word2), 0/1, relations between word1 -> word2

The relations are directional where =>

0  => relation is from premise word -> hypothesis word
1  => relation is from hypothesis word -> premise word

'''

# loop over the dataset
for i in tqdm(range(len(df))):
    prem = df.loc[i, 'premise']
    hyp = df.loc[i, 'hypothesis']

    premwords = clean_sentence(prem).split()
    hypwords = clean_sentence(hyp).split()

    # keep count of relations
    m = 0
    corres = []
    for w1ind, w1 in enumerate(premwords):
        for w2ind, w2 in enumerate(hypwords):
            if((w1, w2) in word_pairs) and w1!=w2 and w1 not in stop_words and w2 not in stop_words:
                m += 1
                rel = comb[(comb['word1'] == w1) & (comb['word2'] == w2)]['Relation'].tolist()
                rel = list(set(rel))
                corres.append(((w1, w1ind), (w2, w2ind), 0, rel))
                
            if((w2, w1) in word_pairs) and w1 != w2 and w1 not in stop_words and w2 not in stop_words:
                m += 1
                rel = comb[(comb['word1'] == w2) & (comb['word2'] == w1)]['Relation'].tolist()
                rel = list(set(rel))
                corres.append(((w2, w2ind), (w1, w1ind), 1, rel))
    

    if m == 0:
        df.loc[i, "Rel_with_toks"] = ""
    else:
        df.loc[i, "Rel_with_toks"] = str(corres)


df.to_csv(save_path, index=False)