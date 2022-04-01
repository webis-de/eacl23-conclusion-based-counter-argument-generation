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

special_tokens_dict = {'additional_special_tokens': ['<conclusion>', '</conclusion>','<premises>', '</premises>', '<counter>']}

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

#Code to prepare the instance for the multi-task learning for weak-premise and counter-generation
def process_wp_instance(instance, tokenizer, max_source_length=512, max_target_length=200, ignore_pad_token_for_loss=True):

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

#Code to evaluate the weak premise multitask generation
def eval_wp_on_validation(model, tokenizer, valid_loader, args):
    
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

def generate_counters(model, tokenizer, data_loader, gen_kwargs, skip_special_tokens=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    generated_attacks = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
            
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
                
            generated_tokens = generated_tokens.cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=skip_special_tokens)
            
            generated_attacks += decoded_preds
    
    return generated_attacks


def evaluate_gen_attacks(generated_attacks, gt_attacks, detailed=False):
    result = {
        'bleu_scores': [],
        'bert-fscores': []
    }

    bertscore_metric = load_metric('bertscore')
    bleu_score = load_metric('bleu')

    bertscore_metric.add_batch(predictions=generated_attacks, references=gt_attacks)    
    bertscore_result = bertscore_metric.compute(lang='en', rescale_with_baseline=True)

    gts = [[nltk.word_tokenize(refs)] if type(refs) != list else [nltk.word_tokenize(ref) for ref in refs] for refs in gt_attacks]
    preds = [nltk.word_tokenize(x) for x in generated_attacks]

    bleu_score.add_batch(predictions=preds, references=gts)    
    bleuscore_result = bleu_score.compute()

    result['bert-fscore'] =np.mean(bertscore_result['f1'])
    result['bleu'] = bleuscore_result['bleu']
    
    if detailed:
        result['bert-fscores'] = bertscore_result['f1']
        result['bleu_scores']  = []
        
        for gt, pred in zip(gt_attacks, generated_attacks):
            #remove empty counters
            gt = [x for x in gt if x != '']
            if len(gt) == 0 or len(pred) == 0:
                print(gt, '---', pred, ' empty..')
                continue
                
            gt  = [nltk.word_tokenize(gt)] if type(gt) != list else [nltk.word_tokenize(ref) for ref in gt]
            pred= nltk.word_tokenize(pred)

            bleu_score.add_batch(predictions=[pred], references=[gt])    
            bluescore_result = bleu_score.compute()

            result['bleu_scores'].append(bluescore_result['bleu'])
        
    return result

#Encoding function for premises and conclusion experiments
def preprocess_function(examples, tokenizer, premises_clm, counter_clm, conclusion_clm=None, conclusion_in_output=False, max_input_length=512, max_target_length=200, conclusion_idx=-1):
    premises   = examples[premises_clm]
    conclusions = examples[conclusion_clm] if conclusion_clm != None else None
    counters = examples[counter_clm]
    
        
    if isinstance(premises[0], list):
        premises = [' '.join(x) for x in premises]
    
    if conclusions is not None and isinstance(conclusions[0], list):
        if conclusion_idx != -1:
            conclusions = ['' if conclusion_idx > len(x)-1 else x[conclusion_idx] for x in conclusions] #just counter an empty conclusion if we don't have anymore conclusions at that required index....
            print(conclusions[0:3])
        else:
            conclusions = [' '.join(x) for x in conclusions]
            print('ddd', conclusions[0:3])

    if conclusions == None or conclusion_in_output== True: #if conclusion is passed and we don't want it in the toutput, then it should be added to the input for the known-conclusion model
        text_inputs = [ '<premises> ' + x + ' </premises>' for x in premises]
    else:
        text_inputs = [ '<conclusion> ' + x[1] + ' </conclusion> ' + ' <premises> ' + x[0] + ' </premises> ' for x in zip(premises, conclusions)]
            
    model_inputs = tokenizer(text_inputs, max_length=max_input_length, truncation=True, padding='max_length')
    
    if isinstance(counters[0], list):
        counters = [' '.join(x) for x in counters]
    
    
    if conclusion_in_output:
        text_outputs = [ '<conclusion> ' + x[0] + ' <counter> ' + x[1]  for x in zip(conclusions, counters)]
    else:
        text_outputs = counters
        
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_outputs, max_length=max_target_length, truncation=True, padding='max_length')    
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    return evaluate_gen_attacks(decoded_preds, decoded_labels)

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

def check_sig(v1s, v2s, alpha=0.05):
    from scipy import stats

    diff = list(map(lambda x1 , x2: x1 - x2, v1s, v2s))
    is_normal = stats.shapiro(diff)[1] > alpha
    
    if is_normal:
        print('Distribution is normal, so using ttest_rel')
        ttest = stats.ttest_rel(v1s, v2s)
        if ttest.statistic >=0:
            if (ttest.pvalue/2) <= alpha:
                return True
            else:
                return False
        else:
            return False

    else:
        print('Distribution is not normal, so using wilcoxon')
        ttest = stats.wilcoxon(v1s, v2s, alternative='greater')
        
        if ttest.statistic >=0:
            if (ttest.pvalue) <= alpha:
                return True
            else:
                return False
        else:
            return False
        

def split_dataframe_per_conc_similarity(df, conc_clm='conclusion', premises_clm='premises'):
    from sentence_transformers import SentenceTransformer, util
    import torch
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Compute semantic similarity
    df['conclusion_embeddings']   = model.encode(df[conc_clm].tolist()).tolist()
    df['premises_embeddings']     = df[premises_clm].apply(lambda x: model.encode(x))
    df['conclusions_in_argument'] = df.apply(lambda x: util.pytorch_cos_sim(torch.tensor(x['conclusion_embeddings']), 
                                                                                          torch.tensor(x['premises_embeddings'])
                                                                    ).tolist()[0], axis=1)
    
    df['conclusions_in_argument'] = df.apply(lambda x: list(zip(x[premises_clm], x['conclusions_in_argument'])), axis=1)
    df['max_sim_to_conclusion'] = df.apply(lambda x: max([x[1] for x in x['conclusions_in_argument']]), axis=1)
    
    df = df.drop(columns=['conclusion_embeddings', 'premises_embeddings', 'conclusions_in_argument'])
    
    return df
        
def remove_similar_sents(df, threshold=0.75, masked_clm='masked_premises'):
    from sentence_transformers import SentenceTransformer, util
    import torch
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Compute semantic similarity
    df['conclusion_embeddings']   = model.encode(df['title'].tolist()).tolist()
    df['premises_embeddings']     = df['post'].apply(lambda x: model.encode(x))
    df['conclusions_in_argument'] = df.apply(lambda x: util.pytorch_cos_sim(torch.tensor(x['conclusion_embeddings']), 
                                                                                          torch.tensor(x['premises_embeddings'])
                                                                    ).tolist()[0], axis=1)
    df['conclusions_in_argument'] = df.apply(lambda x: list(zip(x['post'], x['conclusions_in_argument'])), axis=1)
    
    #filter-out sentences with semantic overlap of more than 0.75
    df['conclusions_in_argument'] = df.apply(lambda x: [x[0] for x in x['conclusions_in_argument'] if x[1] > threshold], axis=1)
    df['num_cand_conc'] = df['conclusions_in_argument'].apply(lambda x: len(x))
    
    df[masked_clm] = df.apply(lambda row: [p for p in row['post'] if p not in row['conclusions_in_argument']] , axis=1)
    df['premises_with_conclusion'] = df.apply(lambda row: row['post'] + ['Therefore, '+row['title']], axis=1)
    
    df = df.drop(columns=['conclusion_embeddings', 'premises_embeddings', 'conclusions_in_argument'])
    
    return df