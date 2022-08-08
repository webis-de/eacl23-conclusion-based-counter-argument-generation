import csv
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from rouge_score import rouge_scorer
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer
from torch.nn import CrossEntropyLoss, MSELoss
import os
from torch.utils.data import DataLoader
import json
import sys
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, source_encodings, conclusions, counter_claims, counters, claim_target_encodings, argument_target_encodings, conclusion_encodings):
        self.source_encodings = source_encodings
        self.conclusions    = conclusions
        self.counter_claims = counter_claims
        self.claim_target_encodings = claim_target_encodings
        self.argument_target_encodings = argument_target_encodings
        self.conclusion_encodings = conclusion_encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.source_encodings.items()}
        item['conclusions']    = self.conclusions[idx]
        item['counter_claims'] = self.counter_claims[idx]
        item['claim_target_encodings'] = torch.tensor(self.claim_target_encodings[idx])
        item['argument_target_encodings'] = torch.tensor(self.argument_target_encodings[idx])
        item['conclusion_encodings'] = torch.tensor(self.conclusion_encodings[idx])

        return item

    def __len__(self):
        return len(self.conclusions)

def parse_df(df_path, max_input_length=512, max_claim_target=100, max_argument_target=256, data_size=None):
    
    if data_size is None:
        df = pd.read_pickle(df_path)
    else:
        df = pd.read_pickle(df_path)[:data_size]

    print("Size of DF: ", len(df))
    conclusions   = df.title.tolist()
    premises      = [x if type(x) == str else ' '.join(x) for x in df.post.tolist()]
    counters      = [x if type(x) == str else ' '.join(x) for x in df.counter.tolist()]
    counter_claims= df.counter_conclusion.tolist()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    source_encodings          = tokenizer(premises, padding='max_length', max_length=max_input_length, truncation=True)
    claim_target_encodings    = tokenizer(counter_claims, padding='max_length', max_length=max_claim_target, truncation=True)['input_ids']
    argument_target_encodings = tokenizer(counters, padding='max_length', max_length=max_argument_target, truncation=True)['input_ids']
    conclusion_encodings      = tokenizer(conclusions, padding='max_length', max_length=max_claim_target, truncation=True)['input_ids']

    ds = Dataset(source_encodings, conclusions, counter_claims, counters, claim_target_encodings, argument_target_encodings, conclusion_encodings)

    return ds