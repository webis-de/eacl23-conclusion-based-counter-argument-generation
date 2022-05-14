from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus    
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
# predict tags for sentences
model = SequenceTagger.load('../../../data-ceph/arguana/arg-generation/claim-target-tagger/model/final-model.pt')

def extract_targets(claims):
    sentences = [Sentence(x) for x in claims]
    model.predict(sentences)
    # iterate through sentences and print predicted labels
    targets = []
    for sentence in sentences:
        target_spans = sorted([(s.text, s.score) for s in sentence.get_spans('ct')], key=lambda x: -x[1])
        if len(target_spans) > 0:
            targets.append(target_spans[0][0])
        else:
            targets.append(sentence.to_original_text())
        
    return targets