import argparse
import logging
from tqdm import tqdm

import numpy as np
import torch
import nltk
import os


from transformers import AdamW
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from datasets import load_dataset, load_metric

from transformers import BartTokenizer, BartForConditionalGeneration
from mt_bart import BartForMultiTaskArgGeneration

from utils import *
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


nltk.download('punkt')

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


def main():

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a argument gen task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
   
    parser.add_argument(
        "--seed", type=str, default=100, help=""
    )
    
    parser.add_argument(
        "--n_gpu", type=str, default=1, help=""
    )
    
    parser.add_argument(
        "--num_epochs", type=int, default=3, help=""
    )
    
    parser.add_argument(
        "--num_beams", type=int, default=3, help=""
    )

    parser.add_argument(
        "--max_source_length", type=int, default=None
    )
        
    parser.add_argument(
        "--max_target_length", type=int, default=None
    )
    
    parser.add_argument(
        "--ignore_pad_token_for_loss",dest='ignore_pad_token_for_loss', action='store_true'
    )
    
    parser.add_argument(
        "--distributed", action='store_true'
    )
    
    parser.add_argument(
        "--train_batch_size", type=int, default=8, 
    )
    
    parser.add_argument(
        "--valid_batch_size", type=int, default=8, 
    )
    
    parser.add_argument(
        "--results_path", type=str
    )
    
    parser.add_argument(
        "--model_path", type=str
    )
    
    parser.add_argument(
        '--tensorboard_log', type=str
    )

    parser.add_argument(
        '--without_classification_head', action='store_true'
    )
    
    parser.add_argument(
        '--wp_weight', type=float, default=0.5
    )
    
    args = parser.parse_args()
    
    set_seed(args)
    
    writer = SummaryWriter(args.tensorboard_log if args.tensorboard_log is not None else args.model_path + '_tensorboard_logs')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForMultiTaskArgGeneration.from_pretrained('facebook/bart-base', without_classification_head=args.without_classification_head, wp_weight=args.wp_weight)

    train_loader, dev_loader = get_data_loaders(args, tokenizer)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=3e-5)

    avg_total_loss = []
    avg_lm_loss = []
    avg_wp_loss = []
    logger.info('----- Start training ------')
    i=1
    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = []
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs=model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, labels = labels, return_dict=True)
            
            avg_total_loss.append(float(outputs.total_loss))
            avg_lm_loss.append(float(outputs.lm_loss))
            avg_wp_loss.append(float(outputs.wp_loss))

            outputs.total_loss.backward(retain_graph=True)
            optim.step()

            if i % 500 == 0:
                total_val_loss, lm_val_loss, wp_val_loss, rouge_results, predictions = eval_on_validation(model, tokenizer, dev_loader, args)
                logger.info('Total training loss: ' + str(outputs.total_loss))
                logger.info('Total valid loss: ' + str(total_val_loss))
                logger.info(rouge_results)
                
                writer.add_scalar("Loss/train", outputs.total_loss, i)
                writer.add_scalar("Loss/valid", total_val_loss, i)
                writer.add_scalar("Rouge/r1", rouge_results['rouge1'], i)
                writer.add_scalar("Rouge/r2", rouge_results['rouge2'], i)
                writer.add_scalar("Rouge/r3", rouge_results['rougeL'], i)
                
            i+=1


        total_val_loss, lm_val_loss, wp_val_loss, rouge_results, predictions = eval_on_validation(model, tokenizer, dev_loader, args)
        logger.info('------------ Results after epoch {} -----------------'.format(epoch))
        logger.info('Total training loss: ' + str(outputs.total_loss))
        logger.info('Total valid loss: ' + str(total_val_loss))
        logger.info(rouge_results)
        logger.info(predictions[0:10])


        model.save_pretrained(args.model_path + "_epoch_%i" % epoch)
        tokenizer.save_pretrained(args.model_path + "_epoch_%i" % epoch)
            
#python run_multitask_arg_gen.py --train_file='../data/train.json' --validation_file='../data/valid.json' --model_path ../data/output/model/v1 --max_source_length 512 --max_target_length 200 --train_batch_size 2 --valid_batch_size 4 --num_epochs 3 2>&1 | tee log_file.log

#For the baseline
#python run_multitask_arg_gen.py --train_file='../data/train.json' --validation_file='../data/valid.json' --model_path ../data/output/model/baseline --max_source_length 512 --max_target_length 200 --train_batch_size 2 --valid_batch_size 4 --without_classification_head --num_epochs 3

if __name__ == "__main__":
    main()
