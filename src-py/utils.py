from collections import defaultdict

import json
import logging
import torch
import nltk
import itertools
import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric


logger = logging.getLogger(__name__)

rouge_metric = load_metric("rouge")
bertscore_metric = load_metric('bertscore')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
#         print(ds['labels'][0])
#         print(ds['input_ids'][0])
#         print(ds['start_positions'][0])
#         print(ds['end_positions'][0])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.ds.items()}
        return item

    def __len__(self):
        return len(self.ds['labels'])

def pad_input_sequence(sequence, tokenizer, padding_token, max_len=None, truncate=False):
    max_len  = max(len(x) for x in sequence) if max_len is None else max_len
    sequence = [s + [padding_token] * (max_len - len(s)) if len(s) <  max_len else s[:max_len]
                for s in sequence]
    
    #adding <s> the cls_token
    #sequence = [s + [tokenizer.sep_token_id] for s in sequence]
    
    #print(sequence[0])
    return sequence

def find_weak_premises_in_arg(argument, weak_premises):
    sen_indices=[]
    for i, s in enumerate(argument):
        s_in_weak_premises = [p.find(s) > -1 or s.find(p) > -1 for p in weak_premises]
        if any(s_in_weak_premises):
            sen_indices.append(i)
    return sen_indices

def process_instance(instance, tokenizer, max_source_length=512, max_target_length=200, ignore_pad_token_for_loss=True):

    processed_instance = {}
    
    argument_sentences = [instance['conclusion']] + instance['premises']
    weak_premises_idxs = find_weak_premises_in_arg(argument_sentences, instance['weak_premises'])
    
    start_positions = []
    end_positions = []
    argument_sentences_tokenized = [tokenizer.tokenize(s) for s in argument_sentences]
    #find start and end position of each of the weak premises
    for weak_premises_idx in weak_premises_idxs:
        start_pos = sum([len(s) for s in argument_sentences_tokenized[0:weak_premises_idx]])
        end_pos   = start_pos + len(argument_sentences_tokenized[weak_premises_idx])
    
        start_positions.append([start_pos + 1]) # +1 because of the sos token
        end_positions.append([end_pos + 1])
    
    #Encode the argument and the counter
    concatenated_argument_sentences = ' '.join(argument_sentences)
    argument_sentences_encoded = tokenizer(concatenated_argument_sentences,  max_length=max_source_length, truncation=True)
    counter_encoded = tokenizer(instance['counter'], max_length=max_target_length, truncation=True)['input_ids']
    
    processed_instance['input_ids'] = argument_sentences_encoded['input_ids']
    processed_instance['attention_mask'] = argument_sentences_encoded['attention_mask']
    processed_instance['labels']    = counter_encoded
    processed_instance['start_positions']  = start_positions[0] #take only one weak premise
    processed_instance['end_positions']  = end_positions[0]

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if ignore_pad_token_for_loss:
        processed_instance["labels"] = [(l if l != tokenizer.pad_token_id else -100) for l in processed_instance['labels']]
        
    #print(instance['weak_premises'][0])
    #print(list(itertools.chain.from_iterable(argument_sentences_tokenized))[start_positions[0]:end_positions[0]])
    
    return processed_instance

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels          

def get_test_loader(dataset_path, tokenizer, args):
    
    test_ds = defaultdict(list)
    logger.info("Parse test dataset")
    raw_ds = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
        num_samples = args.num_test_samples if args.num_test_samples is not None else len(dataset)
        for instance in tqdm(dataset[:num_samples]):
            raw_ds.append(instance)
            processed_instance = process_instance(instance, tokenizer, args)
            test_ds['input_ids'].append(processed_instance['input_ids'])
            test_ds['attention_mask'].append(processed_instance['attention_mask'])
            test_ds['labels'].append(processed_instance['labels'])
            test_ds['start_positions'].append(processed_instance['start_positions'])
            test_ds['end_positions'].append(processed_instance['end_positions'])
            
        for key, value in test_ds.items():
            if key =='labels':
                padding_token=-100
            elif key == 'attention_mask':
                padding_token=0
            else:
                padding_token=tokenizer.pad_token_id
    
        
            if key in ['attention_mask', 'input_ids', 'labels']:
                value = pad_input_sequence(value, tokenizer, padding_token, max_len=(args.max_target_length if key=='labels' else args.max_source_length))
            
            test_ds[key] = torch.tensor(value)
    
        test_ds = Dataset(test_ds)
    
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds) if args.distributed else None
        test_loader  = DataLoader(test_ds, sampler=None, batch_size=args.batch_size, shuffle=False)
        
        return test_loader, raw_ds

def generate_and_evaluate_text(model, tokenizer, data_loader, args, gen_kwargs):
    metric = load_metric('rouge')
    bertscore_metric = load_metric('bertscore')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    generated_attacks = []
    gt_attacks = []
    detected_weak_premises = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
            
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()
            
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id) #To ignore labels that are pad_tokens
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            bertscore_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            
            generated_attacks += decoded_preds
            gt_attacks += decoded_labels
            
        result = metric.compute(use_stemmer=True)
        bertscore_result = bertscore_metric.compute(lang='en', rescale_with_baseline=True)

        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        result['bert-fscore'] = round(np.mean(bertscore_result['f1']), 2)
    
    return result, generated_attacks, gt_attacks, detected_weak_premises

def evaluate_gen_attacks(generated_attacks, gt_attacks):
    metric = load_metric('rouge')
    bertscore_metric = load_metric('bertscore')
    
    metric.add_batch(predictions=generated_attacks, references=gt_attacks)
    bertscore_metric.add_batch(predictions=generated_attacks, references=gt_attacks)
    
    result = metric.compute(use_stemmer=True)
    bertscore_result = bertscore_metric.compute(lang='en', rescale_with_baseline=True)

    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    result['bert-fscore'] = round(np.mean(bertscore_result['f1']), 2)
    
    return result
            
def eval_on_validation(model, tokenizer, valid_loader, args):
    
    metric = load_metric('rouge')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    avg_total_loss = []
    avg_lm_loss = []
    avg_wp_loss = []

    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs=model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, labels = labels, return_dict=True)
            
            avg_total_loss.append(float(outputs.total_loss))
            avg_lm_loss.append(float(outputs.lm_loss))
            avg_wp_loss.append(float(outputs.wp_loss))
            
            #Evalute ROUGE scores
            gen_kwargs = {
                "max_length": args.max_target_length,
                "num_beams": args.num_beams,
            }
            labels = batch["labels"]
            
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()
            
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id) #To ignore labels that are pad_tokens
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            
        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
      
    
    return np.mean(avg_total_loss), np.mean(avg_lm_loss), np.mean(avg_wp_loss), result, decoded_preds

def preprocess_function(examples, tokenizer, input_clm, output_clm, max_input_length=512, max_target_length=200):
    text_inputs = examples[input_clm]
    text_outputs = examples[output_clm]
    
    if isinstance(text_inputs[0], list):
        text_inputs = [' '.join(x) for x in text_inputs]
    
    model_inputs = tokenizer(text_inputs, max_length=max_input_length, truncation=True)
    
    
    if isinstance(text_outputs[0], list):
        text_outputs = [' '.join(x) for x in text_outputs]
        
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_outputs, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """

    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    logger.info("Parse train and validation datasets")
    with open(args.train_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
        for instance in tqdm(dataset):
            processed_instance = process_instance(instance, tokenizer, args)
            datasets['train']['input_ids'].append(processed_instance['input_ids'])
            datasets['train']['attention_mask'].append(processed_instance['attention_mask'])
            datasets['train']['labels'].append(processed_instance['labels'])
            datasets['train']['start_positions'].append(processed_instance['start_positions'])
            datasets['train']['end_positions'].append(processed_instance['end_positions'])


    with open(args.validation_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
        for instance in tqdm(dataset[0:1000]):
            processed_instance = process_instance(instance, tokenizer, args)
            datasets['valid']['input_ids'].append(processed_instance['input_ids'])
            datasets['valid']['attention_mask'].append(processed_instance['attention_mask'])
            datasets['valid']['labels'].append(processed_instance['labels'])
            datasets['valid']['start_positions'].append(processed_instance['start_positions'])
            datasets['valid']['end_positions'].append(processed_instance['end_positions'])

    
    
    logger.info("Build train and validation dataloaders")
    #train_dataset, valid_dataset = Dataset(datasets["train"]), Dataset(datasets["valid"])
    
    for key, value in datasets['train'].items():
        if key =='labels':
            padding_token=-100
        elif key == 'attention_mask':
            padding_token=0
        else:
            padding_token=tokenizer.pad_token_id
    
        if key in ['attention_mask', 'input_ids', 'labels']:
            value = pad_input_sequence(value, tokenizer, padding_token, max_len=(args.max_target_length if key=='labels' else args.max_source_length))
            
        #print(value)
        
        datasets['train'][key] = torch.tensor(value)
        
    for key, value in datasets['valid'].items():
        if key =='labels':
            padding_token=-100
        elif key == 'attention_mask':
            padding_token=0
        else:
            padding_token=tokenizer.pad_token_id
    
        
        if key in ['attention_mask', 'input_ids', 'labels']:
            value = pad_input_sequence(value, tokenizer, padding_token, max_len=(args.max_target_length if key=='labels' else args.max_source_length))
            
        datasets['valid'][key] = torch.tensor(value)
    
    train_dataset, valid_dataset = Dataset(datasets['train']), Dataset(datasets['valid'])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Seq length): {}".format(datasets['train']['input_ids'].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(datasets['valid']['labels'].shape))
    
    return train_loader, valid_loader
