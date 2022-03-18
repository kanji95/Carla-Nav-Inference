import os
import sys
import pandas as pd

from glob import glob

import torch

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

split = sys.argv[1]

data_root = "/ssd_scratch/cvit/kanishk/carla_data/"

command_files = glob(data_root + split + "/*/command.txt")
# print(command_files)

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
)

def check_validity(constituents_):
    word_list = [constituent_['word'] for constituent_ in constituents_ if constituent_["nodeType"] != "CC"]
    count_list = [len(word.split(" ")) for word in word_list]
    if 1 in count_list:
        return False
    else:
        return True

result = {}
for command_file in command_files:
    command = open(command_file, "r").readline()
    command = command.strip().lower()
    if command[-1] == ".":
        command = command[:-1]
    episode = command_file.split("/")[-2]

    predictions = predictor.predict(sentence=command)
    constituents = predictions["hierplane_tree"]["root"]["children"]  # [0]['children']
    if not check_validity(constituents):
        # constituents = predictions["hierplane_tree"]["root"]
        sub_commands = [command]
    else:
        if len(constituents) == 1 and "children" in constituents[0]:
            constituents_ = constituents[0]["children"]
            if not check_validity(constituents_):
                pass
            else:
                constituents = constituents_
            # word_list = [constituent_['word'] for constituent_ in constituents_ if constituent_["nodeType"] != "CC"]
            # count_list = [len(word.split(" ")) for word in word_list]
            # if 1 in count_list:
            #     pass
            # else:
            #     constituents = constituents_
        sub_commands = [
            constituent["word"]
            for constituent in constituents
            if constituent["nodeType"] != "CC"
        ]
    result[int(episode)] = {"command": command, "sub_command": sub_commands}

df1 = pd.DataFrame.from_dict(result, orient="index")
df2 = pd.DataFrame(df1["sub_command"].to_list(), index=df1.index)
df = pd.concat([df1, df2], axis=1)
df.drop("sub_command", axis=1, inplace=True)
df.sort_index(inplace=True)
print(df)

df.to_csv(f'./dataloader/sub_commands_{split}.csv')
