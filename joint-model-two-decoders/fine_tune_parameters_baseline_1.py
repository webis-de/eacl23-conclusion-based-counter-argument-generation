###############
### This script trains a model to generate the conclusion and the counter argument with stance learning
### for only counter vs conclusion
###############
import csv
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import transformers
from rouge_score import rouge_scorer
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.nn import CrossEntropyLoss, MSELoss
import os
from model import MultiTaskBart
from model import OurModel
from utils import parse_df
import time
import sys
import tqdm
import argparse

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(9487) # no different seed

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using GPU? ", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))

data_dir = '../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/'
teacher_model_path='../data/output/stance_classification/best_model/'

#Teacher model
stance_classifier_teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
stance_classifier_teacher_model     = AutoModelForSequenceClassification.from_pretrained(teacher_model_path)


stance_classifier_teacher_model.to(device)
stance_classifier_teacher_model.eval()

def main(args):
    
    
    # NOTE: NEED to write own data loading function
    train_dataset = parse_df(data_dir + 'preprocessed_train_conclusion_all.pkl',
        max_input_length=args.max_input_length, max_claim_target=args.max_claim_target, 
        max_argument_target=args.max_argument_target, data_size=args.train_size)
    dev_dataset   = parse_df(data_dir + 'sample_valid_conclusion_all_preprocessed.pkl',
        max_input_length=args.max_input_length, max_claim_target=args.max_claim_target, 
        max_argument_target=args.max_argument_target, data_size=args.eval_size)

    train_loader= DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, )
    dev_loader= DataLoader(dev_dataset, shuffle=False, batch_size=args.valid_batch_size)
    #test_loader= DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    for alpha1 in [0.1, 0.3, 0.5, 0.7]:
        print('Training with alph1={} and alph2={}'.format(alpha1, 1-alpha1))
        #Our model
        model = OurModel.create(model_name='facebook/bart-large', 
                                model_config=transformers.AutoConfig.from_pretrained('facebook/bart-large'))
        tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-large')

        model.to(device)
        model.train()

        run_training(args, model, tokenizer, train_loader, dev_loader, alpha1, 1-alpha1)

    
          

def run_training(args, model, tokenizer, train_loader, dev_loader, alpha1, alpha2):

    results_path_prefix = args.output_dir + "results"
    models_path_prefix  = args.output_dir + "trained_models"
    if not os.path.exists(results_path_prefix):
        os.makedirs(results_path_prefix, exist_ok=True)
    if not os.path.exists(models_path_prefix):
        os.makedirs(models_path_prefix, exist_ok=True)
       
    
    optim = AdamW(model.parameters(), lr=args.learning_rate)
      

    num_global_steps = 0
    for epoch in range(args.num_epoch):
        start = time.time()
        print('==========start training==========')
        print('Epoch:', epoch)
        i=0

        avg_loss_claim_gen = []
        avg_loss_argument_gen = []
        avg_loss_conc_gen = []
        avg_loss_claim_to_conclusion_stance = []
        avg_loss_argument_to_conclusion_stance = []
        avg_loss_argument_to_claim_stance = []
        avg_loss_conclusion_stance = []

        for batch in  tqdm.tqdm(train_loader):

            optim.zero_grad()

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            claim_targets  = batch['claim_target_encodings'].to(device)
            argument_targets  = batch['argument_target_encodings'].to(device)
            conclusion_targets= batch['conclusion_encodings'].to(device)
            
            batch_conclusions = batch['conclusions']

            batch_conclusion_encoding = tokenizer.batch_encode_plus(batch['conclusions'], return_tensors = "pt", max_length=args.max_input_length, truncation=True, padding=True)
            batch_conclusion_encoding = batch_conclusion_encoding.to(device)

            #1. Running through the conclusion decoder
            conclusion_outputs = model.conclusion_model(input_ids, attention_mask=attention_mask, labels = conclusion_targets, return_dict=True)

            #8. Running through the counter-argument decoder
            counter_argument_outputs = model.counter_argument_model(input_ids, attention_mask=attention_mask, labels = argument_targets, return_dict=True)

            avg_loss_claim_gen.append(0)
            avg_loss_argument_gen.append(float(counter_argument_outputs.loss))
            avg_loss_conc_gen.append(float(conclusion_outputs.loss))
            avg_loss_claim_to_conclusion_stance.append(0)
            avg_loss_argument_to_conclusion_stance.append(float(0))
            avg_loss_argument_to_claim_stance.append(0)
            avg_loss_conclusion_stance.append(float(0))

            loss = alpha1 * conclusion_outputs.loss + alpha2 * counter_argument_outputs.loss
            
            loss.backward()
            optim.step()
            num_global_steps+=1
            i+=1



            if num_global_steps % args.eval_steps == 0:
                print()
                print("--------Start Evaluation--------")
                eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance = run_evaluation(args,  model, tokenizer, dev_loader)
                avg_eval_loss = sum([eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance])/len([eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance])
                print('Validation losses: {}, {}, {}, {}, {}, {}, {}, {}'.format(avg_eval_loss, eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance))

                with open(results_path_prefix +'/training_log.txt', 'a') as file:
                    file.write('\n')
                    file.write("Parameters: {}, {} ".format(alpha1, alpha2))
                    file.write("Epoch " + str(epoch) + " Step " + str(num_global_steps))
                    file.write('\n')
                    file.writelines(['Validation losses: {}, {}, {}, {}, {}, {}, {}, {}'.format(avg_eval_loss, eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance), '\n\n\n'])

                
                path_model = models_path_prefix + "/models-global-step-" + str(num_global_steps) + "-" + "{}-{}".format(alpha1, alpha2)
                model.save_model(path_model)
                tokenizer.save_pretrained(path_model)


def run_evaluation(args, model, tokenizer, dev_loader):
    model.eval()

    with torch.no_grad():
        
        avg_loss_claim_gen = []
        avg_loss_argument_gen = []
        avg_loss_claim_to_conclusion_stance = []
        avg_loss_argument_to_conclusion_stance = []
        avg_loss_argument_to_claim_stance = []
        avg_loss_conc_gen = []
        avg_loss_conclusion_stance = []

        for batch in tqdm.tqdm(dev_loader):

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            claim_targets  = batch['claim_target_encodings'].to(device)
            argument_targets  = batch['argument_target_encodings'].to(device)
            conclusion_targets= batch['conclusion_encodings'].to(device)
            
            batch_conclusions = batch['conclusions']

            batch_conclusion_encoding = tokenizer.batch_encode_plus(batch['conclusions'], return_tensors = "pt", max_length=args.max_input_length, truncation=True, padding=True)
            batch_conclusion_encoding = batch_conclusion_encoding.to(device)

            #1. Running through the conclusion decoder
            conclusion_outputs = model.conclusion_model(input_ids, attention_mask=attention_mask, labels = conclusion_targets, return_dict=True)

            #8. Running through the counter-argument decoder
            counter_argument_outputs = model.counter_argument_model(input_ids, attention_mask=attention_mask, labels = argument_targets, return_dict=True)

            avg_loss_claim_gen.append(0)
            avg_loss_argument_gen.append(float(counter_argument_outputs.loss))
            avg_loss_conc_gen.append(float(conclusion_outputs.loss))
            avg_loss_claim_to_conclusion_stance.append(0)
            avg_loss_argument_to_conclusion_stance.append(float(0))
            avg_loss_argument_to_claim_stance.append(0)
            avg_loss_conclusion_stance.append(float(0))


    model.train()
    return  np.mean(avg_loss_claim_gen), np.mean(avg_loss_argument_gen), np.mean(avg_loss_conc_gen), \
    np.mean(avg_loss_claim_to_conclusion_stance), np.mean(avg_loss_argument_to_conclusion_stance), np.mean(avg_loss_argument_to_claim_stance),\
    np.mean(avg_loss_conclusion_stance)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a argument gen task")

    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--max_input_length', type=int, default=325)
    parser.add_argument('--max_claim_target', type=int, default=64)
    parser.add_argument('--max_argument_target', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=8)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--eval_size', type=int, default=None)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--eval_steps', type=int, default=1000)
    
    args = parser.parse_args()

    main(args)
