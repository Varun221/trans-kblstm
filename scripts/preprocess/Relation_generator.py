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
from torchaudio import save_encinfo
import transformers
from sklearn.model_selection import train_test_split

import time
import copy
from tqdm import tqdm

import transformers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse

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
    
    for ch in ")(:-.,°′`\"\'%$+;/\#&?!_":
        line = line.replace(ch, f" {ch} ")

    return line


# get knowledge database
comb = pd.read_csv("../../data/knowledge_db/Full_Knowledge_db.csv")

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
    'dbpedia/product': 'has product'
    
}

Rels_list = list(Rel_mapping.keys())

word_pairs = set(list(zip(comb['word1'], comb['word2'])))

for i in tqdm(range(len(df))):
    prem = df.loc[i, 'premise']
    hyp = df.loc[i, 'hypothesis']

    premwords = clean_sentence(prem).split()
    hypwords = clean_sentence(hyp).split()

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