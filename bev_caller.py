import os

datasets = ['val']

for dataset in datasets:
    if dataset == 'val':
        episodes = list(range(0, 25))
        episodes = [22]
    elif dataset == 'test':
        episodes = list(range(0, 34))
        episodes = [28]
    else:
        raise NotImplementedError(
            f'infer_dataset {dataset}: episode counts not set')
    for i in episodes:
        os.system(
            f"python get_bev.py --infer_dataset {dataset} --spawn {i} --sync")
