import sys
sys.path.insert(0, '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/')

import torch
import pandas as pd
import numpy as np
import statistics
import torch
import random
import time
import numpy as np
#from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import shutil
import itertools

from model.GRU_model import *
from pipeline_config import *
from window_size_utils import *

# Seeds for comparisons:

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

with open(train_dataset_path, 'rb') as f:
    train_dataset_dict = pickle.load(f)
    
with open(test_dataset_path, 'rb') as f:
    test_dataset_dict = pickle.load(f)

with open(category_id_to_category_name_path, 'rb') as f:
    category_id_to_name_dict = pickle.load(f)
    
with open(categories_per_indent_path, 'rb') as f:
    categories_per_indent_dict = pickle.load(f)

categories_lists = list(categories_per_indent_dict.values())
categories_id = list(itertools.chain.from_iterable(categories_lists))
categories = []
for category_id in categories_id:
    categories.append(category_id_to_name_dict[category_id])

def pipline(train_dataset_dict, test_dataset_dict):
    results = {}
    #for category in categories:
    for category in ['All items']:
        train_dataloader, test_dataloader = create_dataloader(train_dataset_dict[category], test_dataset_dict[category])
        #print(f'train_data: {train_dataset_dict[category]}')
        #print(f'test data: {test_dataset_dict[category]}')
        model = Model
        model.to(device)
        
        optimizer = Optimizer

        parameters_file_name = category+'.pt'
        
        #for name, param in model.named_parameters():
        #    print(f"{name}: {param}")
        results[category] = training_and_evaluation(
                                model=model,
                                optim=optimizer,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                category=category,
                                checkpoint_path=CheckpointPath+parameters_file_name,
                                best_checkpoint_path=BestcheckpointPath+parameters_file_name,
                            )
    
    return results

    results = pipline(train_dataset_dict, test_dataset_dict)

    with open('/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/basic/model_results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    best_predictions_dict = get_best_predictions_for_each_category(best_models_dict, train_dataset_dict, test_dataset_dict)

with open('/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/basic/predictions_dict.pickle', 'wb') as handle:
    pickle.dump(best_predictions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

dir_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/basic/checkpoints/best_checkpoints/'

category_id_list = []
 
# list out keys and values separately
key_list = list(category_id_to_name_dict.keys())
val_list = list(category_id_to_name_dict.values())

for cat_name in categories:
    position = val_list.index(cat_name)
    category_id_list.append(key_list[position])

weights_dict = get_weights_per_category(category_id_list, category_id_to_name_dict, dir_path)

with open(sgru_model_weights_path, 'wb') as handle:
    pickle.dump(weights_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)