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

split = sys.argv[1]
            
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

predictor = load_predictor("structured-prediction-constituency-parser")

df_command = pd.read_csv(f"./dataloader/sub_commands_{split}.csv", index_col=0)
print(df_command.shape)

data_path = f'/ssd_scratch/cvit/kanishk/carla_data/{split}/'

results = {}
for episode in range(df_command.shape[0]):
    
    # vehicle_path = data_path + str(episode) + "/vehicle_positions.txt"
    target_path = data_path + str(episode) + "/target_positions.txt"
    
    # vehicle_positions = pd.read_csv(vehicle_path, names=["x", "y", "z"])
    target_positions = pd.read_csv(target_path, names=["x", "y", "z", "click_no"])
    
    final_click_idx = target_positions["click_no"].max()
    
    
    command = df_command.loc[episode]['command']
    
    preds = predictor.predict(command)
    hierplane_tree = preds['hierplane_tree']
    
    sub_phrases = []
    tree_traversal(hierplane_tree['root'], sub_phrases)
    
    while command in sub_phrases:
        sub_phrases.remove(command)
        
    sub_command_0 = df_command.loc[episode]['sub_command_0']
    sub_command_1 = df_command.loc[episode]['sub_command_1'] if not pd.isna(df_command.loc[episode]['sub_command_1']) else df_command.loc[episode]['sub_command_0']
    sub_commands = [sub_command_0, sub_command_1]
    
    distance_0 = [0]*len(sub_phrases)
    for idx, phrase in enumerate(sub_phrases):
        distance_0[idx] = sentence_distance(phrase, sub_command_0)
    distance_0 = torch.tensor(distance_0)
    
    distance_1 = [0]*len(sub_phrases)
    for idx, phrase in enumerate(sub_phrases):
        distance_1[idx] = sentence_distance(phrase, sub_command_1)
    distance_1 = torch.tensor(distance_1)
    
    distances = [distance_0, distance_1]
    
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

    sub_command_list = []
    similarity_gt_list = []
    
    for frame_no in range(target_positions.shape[0]):
        
        if target_positions.loc[frame_no]['click_no'] == final_click_idx:
            sub_command = sub_commands[-1]
            distance = distances[-1]
        else:
            sub_command = sub_commands[0]
            distance = distances[0]
        
        similarity_gt = torch.ones(len(distance)) * -1
        similarity_gt[distance.argmin()] = 1

        sub_command_list.append(sub_command)
        similarity_gt_list.append(similarity_gt)
        
    results[episode] = {
        "sub_phrases": sub_phrases,
        "tree_embedding": torch.split(last_hidden_states, 1, dim=0),
        "attention_mask": attention_mask,
        "frame_sub_command": sub_command_list,
        "frame_similarity_gt": similarity_gt_list
    }    

torch.save(results, f"./dataloader/{split}_tree_embeddings.pt")
