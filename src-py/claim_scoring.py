import os
import sklearn
import html
import sys
import numpy as np
import torch
import re
import argparse
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from fastai.text.all import *
from fastai.learner import load_learner

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def load_lm_data(data_path):
    texts = []
    for line in open(data_path, encoding='utf-8'):
        texts.append(line.strip())

    col_names = ['labels','text']

    df_trn = pd.DataFrame({'text':texts, 'labels':[0]*len(texts)}, columns=col_names)

    df_trn['text'] = df_trn['text'].apply(lambda x: fixup(x))

    return df_trn

def load_classifier_data(data_path):
    texts  = []
    labels = [] 
    for line in open(data_path, encoding='utf-8'):

        text_label = line.strip().split('\t')
        if len(text_label) > 1:
            texts.append(text_label[0])
            labels.append(int(text_label[1]))

    df = pd.DataFrame({'text':texts, 'labels':labels}, columns=['labels','text'])
    df['text'] = df['text'].apply(lambda x: fixup(x))

    return df


class ClaimMiner(object):
    def __init__(self, model_path, scenario='Training'):
        self.model_path = model_path
        
        if scenario == 'Prediction':
            self.learner = load_learner(self.model_path)
        else:
            self.learner = None

        self.data_lm = None

    def train_lm(self, data_path):
        df_train = load_lm_data(data_path)
        df_train = df_train.sample(2000000)
        self.data_lm = TextDataLoaders.from_df(df = df_train, is_lm=True, valid_pct=0.2,text_col=1, label_col=0, path = self.model_path)
        
        learn = language_model_learner(self.data_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], 
                                       path=self.model_path, pretrained = True, wd=0.1).to_fp16()
        learn.fit_one_cycle(1, 1e-2)
        learn.save('1epoch')

        learn.unfreeze()
        learn.fit_one_cycle(3, 1e-3)

        learn.save_encoder('finetuned_encoder')
        torch.save(self.data_lm, self.model_path + '/data_lm.pkl')


    def train_classifier(self, data_path, epochs=10):
        self.data_lm = torch.load(self.model_path + '/data_lm.pkl')
        
        classification_df = load_classifier_data(data_path)
        
        dls_clas = TextDataLoaders.from_df(df=classification_df, text_col=1, label_col=0, valid_pct=0.2, text_vocab=self.data_lm.vocab)
        
        learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy, path=self.model_path)
        learn = learn.load_encoder('finetuned_encoder')
        
        learn.fit_one_cycle(1, 2e-2)
        
        learn.freeze_to(-2)
        learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
        
        learn.freeze_to(-3)
        learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

        learn.unfreeze()
        learn.fit_one_cycle(epochs, slice(1e-3/(2.6**4),1e-3))
        
        learn.export()

    def predict(self, texts):
        if self.learner == None:
            self.learner = load_learner(self.model_path)

        preds = [self.learner.predict(text)[2] for text in texts]

        return preds



def get_claim(cminer, text, topic):
    sents = sent_tokenize(text)
    #filter only sentences that overlap with the topic...
    sents_preds   = [cminer.learner.predict(sent) for sent in sents]
    
    sents_out = [(x[0], (x[1][2][0].item(), x[1][2][1].item())) for x in zip(sents, sents_preds)]
    sents_out = sorted(sents_out, key=lambda x: -x[1][1])
    
    return sents_out

def choose_claim(claims_scores, min_len=15):
    if len(claims_scores) > 0:
        #filter out short sentences
        claims_scores = [x for x in claims_scores if len(x[0].split(' ')) > min_len]
        if len(claims_scores) == 0:
            return ''
        else:
            return sorted(claims_scores, key=lambda x: -x[1][1])[0][0]
    else:
        return ''

def mine_claim_from_df(df_path, cminer_path, output_path):
    #'/workspace/ceph_data/belief-based-argumentation-generation/models/arg_mining'
    cminer = ClaimMiner(cminer_path)
    df = pd.read_csv(df_path)

    df['claims'] = df.apply(lambda row: get_claim(cminer, row['opinion_txt'], row['topic']), axis=1)
    df['top_claim'] = df.claims.apply(lambda x: choose_claim(x))

    df.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument mining')
    parser.add_argument('--data_frame_path')
    parser.add_argument('--output_path')
    parser.add_argument('--cminer_path')

    args = parser.parse_args()

    mine_claim_from_df(args.data_frame_path, args.cminer_path, args.output_path)

