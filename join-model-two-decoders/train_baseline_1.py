###############
### This script trains a model to generate the conclusion and the counter argument without stance learning
### for both conclusion and counter argument
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
teacher_model_path='../multitask-counter-arg-generation/data/output/stance_classification/best_model/'


#Teacher model
stance_classifier_teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
stance_classifier_teacher_model     = AutoModelForSequenceClassification.from_pretrained(teacher_model_path)

#Our model
model = OurModel.create(model_name='facebook/bart-large', 
                        model_config=transformers.AutoConfig.from_pretrained('facebook/bart-large'))
tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-large')

model.to(device)
model.train()

stance_classifier_teacher_model.to(device)
stance_classifier_teacher_model.eval()


def main(args):
    
    # NOTE: all parameters are here
    # TODO: use arg input
        
   
    alpha1 = 1
    alpha2 = 0.6
    alpha3 = 0.9
    
    
    # NOTE: NEED to write own data loading function
    train_dataset = parse_df(data_dir + 'preprocessed_train_conclusion_all.pkl',max_input_length=args.max_input_length, max_claim_target=args.max_claim_target, max_argument_target=args.max_argument_target)
    dev_dataset   = parse_df(data_dir + 'sample_valid_conclusion_all_preprocessed.pkl',max_input_length=args.max_input_length, max_claim_target=args.max_claim_target, max_argument_target=args.max_argument_target)

    train_loader= DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, )
    dev_loader= DataLoader(dev_dataset, shuffle=False, batch_size=args.valid_batch_size)
    #test_loader= DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    

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
        avg_loss_claim_to_conclusion_stance = []
        avg_loss_argument_to_conclusion_stance = []
        avg_loss_argument_to_claim_stance = []
        avg_loss_conclusion_stance = []
        avg_loss_conc_gen = []

        model.train()
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

            loss = conclusion_outputs.loss + alpha2 * counter_argument_outputs.loss
            
            loss.backward()
            optim.step()
            num_global_steps+=1
            i+=1


            if num_global_steps % args.eval_steps == 0:
                print()
                print("--------Start Evaluation--------")
                eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance = run_evaluation(args, dev_loader)
                print('Validation losses: {}, {}, {}, {}, {}, {}, {}'.format(eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance))

                with open(results_path_prefix +'/training_log.txt', 'a') as file:
                    file.write('\n')
                    file.write("Epoch " + str(epoch) + " Step " + str(num_global_steps))
                    file.write('\n')
                    file.writelines([#'Source: ', tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=False), '\n\n', 
                                     #'Consluion Target: ', generated_conclusion[0], '\n\n', 
                                     #'Argument Target: ', generated_counter_argument[0], '\n\n',
                                     'Validation losses: {}, {}, {}, {}, {}, {}, {}'.format(eval_loss_claim_gen, eval_loss_argument_gen, eval_loss_conc_gen, eval_loss_claim_to_conclusion_stance, eval_loss_argument_to_conclusion_stance, eval_loss_argument_to_claim_stance, eval_loss_conclusion_stance), '\n\n\n'])

                
                path_model = models_path_prefix + "/models-global-step-" + str(num_global_steps)
                model.save_model(path_model)
                tokenizer.save_pretrained(path_model)

    print('training done')
                    

def run_evaluation(args, dev_loader):
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


    return  np.mean(avg_loss_claim_gen), np.mean(avg_loss_argument_gen), np.mean(avg_loss_conc_gen), np.mean(avg_loss_claim_to_conclusion_stance), np.mean(avg_loss_argument_to_conclusion_stance), np.mean(avg_loss_argument_to_claim_stance), np.mean(avg_loss_conclusion_stance)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a argument gen task")

    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--max_input_length', type=int, default=325)
    parser.add_argument('--max_claim_target', type=int, default=64)
    parser.add_argument('--max_argument_target', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eval_steps', type=int, default=1000)
    
    args = parser.parse_args()

    main(args)