from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset, load_metric, Dataset
import torch
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
tokenizer = AutoTokenizer.from_pretrained("../../../data-ceph/arguana/arg-generation/conclusion-generation-models/dbart")
model = AutoModelForSeq2SeqLM.from_pretrained("../../../data-ceph/arguana/arg-generation/conclusion-generation-models/dbart").to(device)

def truncate_text(text, remove_extra_tokens=0):
        for i in range(3):
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            ).input_ids
            max_model_length = tokens.size()[1]
            truncated_tokens = tokens[0][: max_model_length - remove_extra_tokens]
            text = tokenizer.decode(
                truncated_tokens, clean_up_tokenization_spaces=True
            )
            without_truncate_length = tokenizer(
                text, return_tensors="pt"
            ).input_ids.size()[1]
            if max_model_length > without_truncate_length:
                return tokens, text
        return truncate_text(text, remove_extra_tokens=remove_extra_tokens + 5)

def generate_conclusion(premises, gen_kwargs, batch_size=16):
    if type(premises[0]) == list:
        premises = [' '.join(x) for x in premises]
    
    ds = Dataset.from_dict({'premises': premises})
    ds = ds.map(lambda x :tokenizer(x['premises'], max_length=512, truncation=True, padding='max_length') , batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    generated_conclusion = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            generated_conclusion += decoded_preds

    return generated_conclusion