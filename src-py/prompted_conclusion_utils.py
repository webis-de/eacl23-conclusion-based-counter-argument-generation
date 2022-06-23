import os
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers.generation_logits_process import * 
from datasets import load_dataset, load_metric, Dataset
import torch
from project_debater_api import *
import pandas as pd
from tqdm import tqdm
import spacy

tqdm.pandas()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def get_wiki_concepts(posts, min_len=1, max_concepts=3):
    post_concepts = []
    for post in posts:
        wiki_concepts = term_wikifier(post)
        wiki_concepts = filter(lambda x: len(x[0].split()) > min_len and x[1] > 0, wiki_concepts)
        wiki_concepts = sorted(list(set(wiki_concepts)),key=lambda x: -x[1])
        #take top max_concepts
        post_concepts.append([x[0] for x in wiki_concepts[:max_concepts]])
        
    return post_concepts


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, torch.finfo(scores.dtype).min)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                enforced_vocab = self._prefix_allowed_tokens_fn(batch_id, sent)
                if enforced_vocab != None:
                    mask[batch_id * self._num_beams + beam_id, enforced_vocab] = 0
                else: #if the prefix_allowed_tokens_fn doesn't offer any restriction then apply no restriction..
                    mask[batch_id * self._num_beams + beam_id, :] = 0

        return scores + mask

def allowed_prefix(prefixs, tokenizer, batch_id, sent):
    prefix     = prefixs[batch_id]
    gen_tokens = sent[1:] #skip the bos token
    encoded_pref = tokenizer.encode(prefix, add_special_tokens=False)
    decoded_sent = tokenizer.decode(gen_tokens)
    
    remained_tokens = [x for x in encoded_pref if x not in gen_tokens]
    
    # print(gen_tokens)
    # print(decoded_sent)
    # print(encoded_pref)
    # print('remained:', remained_tokens)
    
    if len(remained_tokens) > 0: #just control the first token
        return [remained_tokens[0]]
    else:
        None
        

def generate_prompted_conclusions(model, tokenizer, premises, prefixes, gen_kwargs, batch_size=16):

    if type(premises[0]) == list:
        premises = [' '.join(x) for x in premises]
    
    ds = Dataset.from_dict({'premises': premises, 'prefixes': prefixes})
    ds = ds.map(lambda x :tokenizer(x['premises'], max_length=512, truncation=True, padding='max_length') , batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    generated_conclusion = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            batch_prefixs = batch['prefixes']
            processors = LogitsProcessorList()
            processors.append(PrefixConstrainedLogitsProcessor(
                lambda batch_id, sent: allowed_prefix(batch_prefixs, tokenizer, batch_id, sent), gen_kwargs['num_beams']))
    
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                logits_processor = processors,
                **gen_kwargs
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            generated_conclusion += decoded_preds

    return generated_conclusion


def generate_prompted_counters(model, tokenizer, premises, prefixes, gen_kwargs, batch_size=16):

    if type(premises[0]) == list:
        premises = [' '.join(x) for x in premises]
    
    ds = Dataset.from_dict({'premises': premises, 'prefixes': prefixes})
    ds = ds.map(lambda x :tokenizer(x['premises'], max_length=512, truncation=True, padding='max_length') , batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    generated_counters = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            batch_prefixs = batch['prefixes']
            processors = LogitsProcessorList()
            processors.append(PrefixConstrainedLogitsProcessor(
                lambda batch_id, sent: allowed_prefix(batch_prefixs, tokenizer, batch_id, sent), gen_kwargs['num_beams']))
    
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                logits_processor = processors,
                **gen_kwargs
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

            generated_counters += decoded_preds

    return generated_counters

def generate_two_seq_prompted_counters(model, tokenizer, premises, prefixes, conclusion_gen_kwargs, argument_gen_kwargs, batch_size=16):

    if type(premises[0]) == list:
        premises = [' '.join(x) for x in premises]
    
    ds = Dataset.from_dict({'premises': premises, 'prefixes': prefixes})
    ds = ds.map(lambda x :tokenizer(x['premises'], max_length=512, truncation=True, padding='max_length') , batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    generated_counter_arguments = []
    generated_conclusions = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_prefixs = batch['prefixes']

            processors1 = LogitsProcessorList()
            processors1.append(PrefixConstrainedLogitsProcessor(
                lambda batch_id, sent: allowed_prefix(batch_prefixs, tokenizer, batch_id, sent), conclusion_gen_kwargs['num_beams']))
    
        
            processors2 = LogitsProcessorList()
            processors2.append(PrefixConstrainedLogitsProcessor(
                lambda batch_id, sent: allowed_prefix(batch_prefixs, tokenizer, batch_id, sent), argument_gen_kwargs['num_beams']))
            
            generated_conclusion_tokens = model.generate_conclusion(input_ids, attention_mask, conclusion_gen_kwargs, processors1) 
            generated_argument_tokens   = model.generate_counter_argument(input_ids, attention_mask, argument_gen_kwargs, processors2)
            
            if isinstance(generated_conclusion_tokens, tuple):
                generated_conclusion_tokens = generated_conclusion_tokens[0]
                
            if isinstance(generated_argument_tokens, tuple):
                generated_argument_tokens = generated_argument_tokens[0]
            
            generated_argument_tokens = generated_argument_tokens.cpu().numpy()
            decoded_arguments = tokenizer.batch_decode(generated_argument_tokens, skip_special_tokens=True)
            
            generated_conclusion_tokens = generated_conclusion_tokens.cpu().numpy()
            decoded_conclusions = tokenizer.batch_decode(generated_conclusion_tokens, skip_special_tokens=True)
            
            generated_counter_arguments +=decoded_arguments
            generated_conclusions +=decoded_conclusions

    return generated_conclusions, generated_counter_arguments