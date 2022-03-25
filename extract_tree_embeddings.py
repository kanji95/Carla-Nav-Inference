import sys
import torch

import numpy as np
import pandas as pd

from nltk.tree import *

from transformers import DistilBertTokenizer, DistilBertModel
from allennlp_models.pretrained import load_predictor

from pyxdameraulevenshtein import damerau_levenshtein_distance

def sentence_distance(sent1, sent2):
    words1 = sent1.split()
    words2 = sent2.split()
    
    return damerau_levenshtein_distance(words1, words2)

def tree_traversal(root, sub_phrases):
    if not 'children' in root:
        if root['nodeType'] == 'VB' and len(root['word'].split()):
            sub_phrases.append(root['word'])
        return
    else:
        # print("* " + root['word'] + ", " + str(root['nodeType']))
        sub_phrases.append(root['word'])
        for node in root['children']:
            tree_traversal(node, sub_phrases)

split = "val" # sys.argv[1]
            
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

predictor = load_predictor("structured-prediction-constituency-parser")

df = pd.read_csv(f"./dataloader/sub_commands_{split}.csv", index_col=0)
print(df.shape)

results = {}
for step, sentence in enumerate(df['command']):

    preds = predictor.predict(sentence)

    hierplane_tree = preds['hierplane_tree']
    
    sub_phrases = []
    tree_traversal(hierplane_tree['root'], sub_phrases)
    
    while sentence in sub_phrases:
        sub_phrases.remove(sentence)
    print(f'{step}: {sub_phrases}')
    
    distance = [0]*len(sub_phrases)
    for idx, phrase in enumerate(sub_phrases):
        distance[idx] = sentence_distance(phrase, sentence)
    distance = torch.tensor(distance)
    print(distance)
    
    ground_truth = torch.zeros(len(distance))
    ground_truth[distance.argmin()] = 1
    print(ground_truth)
    
    encoded = tokenizer(
        text=sub_phrases,
        add_special_tokens=True,
        max_length=15,
        pad_to_max_length=True,
        return_attention_mask = True,
        return_tensors = 'pt',
        truncation=True
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    outputs = model(**encoded)
    
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
    
    results[step] = {
        'sub_phrases': sub_phrases,
        "embedding": torch.split(last_hidden_states, 1, dim=0),
        "ground_truth": ground_truth
    }
    

torch.save(results, f"./dataloader/{split}_tree_embeddings.pt")