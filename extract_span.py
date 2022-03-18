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
# predictor.eval()

result = {}
for command_file in command_files:
    command = open(command_file, "r").readline()
    command = command.strip().lower()
    if command[-1] == ".":
        command = command[:-1]
    episode = command_file.split("/")[-2]
    # print(command, episode)

    predictions = predictor.predict(sentence=command)
    constituents = predictions["hierplane_tree"]["root"]["children"]  # [0]['children']

    if len(constituents) == 1 and "children" in constituents[0]:
        constituents = constituents[0]["children"]
    # print(constituents)
    sub_commands = [
        constituent["word"]
        for constituent in constituents
        if constituent["nodeType"] != "CC"
    ]
    # print(sub_commands)
    result[int(episode)] = {"command": command, "sub_command": sub_commands}

# import pdb; pdb.set_trace()
df1 = pd.DataFrame.from_dict(result, orient="index")
df2 = pd.DataFrame(df1["sub_command"].to_list(), index=df1.index)
df = pd.concat([df1, df2], axis=1)
df.drop("sub_command", axis=1, inplace=True)
df.sort_index(inplace=True)
print(df)

df.to_csv(f'./dataloader/sub_commands_{split}.csv')
