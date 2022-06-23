from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, pipeline, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, TextClassificationPipeline, Seq2SeqTrainer, BertForSequenceClassification
from datasets import load_dataset, load_metric, Dataset
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from transformers import BartTokenizer, BartForConditionalGeneration

import matplotlib.pyplot as plt

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus    
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

###### Loading conclusion generation model ##############
conclusion_gen_tokenizer = AutoTokenizer.from_pretrained("../data/output/conc-gen-model/")
conclusion_gen_model = AutoModelForSeq2SeqLM.from_pretrained("../data/output/conc-gen-model/").to(device)

###### Loading the stance-classification model #########
stance_class_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
stance_class_model = AutoModelForSequenceClassification.from_pretrained('../data/output/stance_classification/best_model').cuda()
arg_stance_pipeline = TextClassificationPipeline(model=stance_class_model, tokenizer=stance_class_tokenizer, framework='pt', task='ArgQ', device=0)

####### Loading the argument quality pipeline ###########
gretz_model = BertForSequenceClassification.from_pretrained('../../data-ceph/arguana/arg-generation/argument-quality/argument-quality-model/checkpoint-9000', local_files_only=True, cache_dir='cache')
gretz_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
arg_quality_pipeline = TextClassificationPipeline(model=gretz_model, tokenizer=gretz_tokenizer, framework='pt', task='ArgQ', device=0)

###### Loading claim target identifier ############   
target_identifier_model = SequenceTagger.load('../../data-ceph/arguana/arg-generation/claim-target-tagger/model/final-model.pt')

def extract_targets(claims):
    sentences = [Sentence(x) for x in claims]
    target_identifier_model.predict(sentences)
    # iterate through sentences and print predicted labels
    targets = []
    for sentence in sentences:
        target_spans = sorted([(s.text, s.score) for s in sentence.get_spans('ct')], key=lambda x: -x[1])
        if len(target_spans) > 0:
            targets.append(target_spans[0][0])
        else:
            targets.append(sentence.to_original_text())
        
    return targets

def get_arg_quality(sents):
    # set the pipeline
    results = arg_quality_pipeline(sents, truncation=True)
    scores = [round(s['score'], 2) for s in results]
    return round(np.mean(scores), 2), scores

def get_stance_scores(sents1, sents2):
    #compute stance score using our trained model
    text_inputs = [x[0] + ' </s> ' + x[1] for x in zip(sents1, sents2)]
    #print(text_inputs)
    stance_results = arg_stance_pipeline(text_inputs, truncation=True)
    stance_labels = [int(x['label'].split('_')[-1]) for x in stance_results]
    stance_scores = [x['score'] for x in stance_results]
    return sum(stance_labels)/len(stance_labels), stance_labels, stance_scores  #The score is the percentage of cases we generated a counter
    
def generate_conclusion(premises, gen_kwargs, batch_size=16):
    if type(premises[0]) == list:
        premises = [' '.join(x) for x in premises]
    
    ds = Dataset.from_dict({'premises': premises})
    ds = ds.map(lambda x :conclusion_gen_tokenizer(x['premises'], max_length=512, truncation=True, padding='max_length') , batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    generated_conclusion = []

    conclusion_gen_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            generated_tokens = conclusion_gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = conclusion_gen_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            generated_conclusion += decoded_preds

    return generated_conclusion