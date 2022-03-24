import torch
from nltk.tree import *
from allennlp_models.pretrained import load_predictor

predictor = load_predictor("structured-prediction-constituency-parser")

# sentence = [["make a right and then stop near the bus stand"], ["take a right and park near blue car"]]

sentence = "make a right and then stop near the bus stand"

preds = predictor.predict(sentence)
print(preds.keys())

for key in preds:
    print(key, type(preds[key]))
    if isinstance(preds[key], list):
        if not isinstance(preds[key][0], str):
            preds[key] = torch.tensor(preds[key]) #[None]
    elif isinstance(preds[key], int):
        preds[key] = torch.tensor(preds[key]) #[None]
        
cp_tree = Tree.fromstring(preds['trees'])
print(cp_tree)

# hrf = predictor._model.make_output_human_readable(preds)
# print(hrf)

# import benepar, spacy
# nlp = spacy.load('en_core_web_md')
# nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
# doc = nlp('The time for action is now. It is never too late to do something.')
# sent = list(doc.sents)[0] 
# print(sent._.parse_string)