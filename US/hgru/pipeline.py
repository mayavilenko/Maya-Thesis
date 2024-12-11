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
torch.use_deterministic_algorithms(True)

with open(train_dataset_path, 'rb') as f:
    train_dataset_dict = pickle.load(f)
    
with open(test_dataset_path, 'rb') as f:
    test_dataset_dict = pickle.load(f)

with open(category_id_to_category_name_path, 'rb') as f:
    category_id_to_name_dict = pickle.load(f)
    
with open(categories_per_indent_path, 'rb') as f:
    categories_per_indent_dict = pickle.load(f)

with open(son_parent_path, 'rb') as f:
    son_parent_dict = pickle.load(f)


def hgru_model(son_parent_dict, train_dataset_dict, test_dataset_dict, categories_per_indent_dict, category_id_to_name_dict, weights_path):
    hgru_models = {}

    for indent in sorted(list(categories_per_indent_dict.keys())):
        print(indent)
        for category in categories_per_indent_dict[indent]:
            category_name = category_id_to_name_dict[category]

            if int(indent) == 0 or son_parent_dict[category] not in categories_per_indent_dict[indent-1]:
                #print(f'category with 0 loss coef: {category_name}|{category}')
                loss_coef=0
                parent_weights=0
            else:
                son = category
                parent = son_parent_dict[son]
                parent_name = category_id_to_name_dict[parent]
                loss_coef = 0.09103560997886097 #post optuna
                parent_model = Model
                parent_optimizer = Optimizer
                parent_model, optimizer, checkpoint, valid_loss_min = load_checkpoint(weights_path+parent_name+'.pt', parent_model, parent_optimizer)
                parent_weights = unify_model_weights(parent_model)

            train_dataloader, test_dataloader = create_dataloader(train_dataset_dict[category_name], test_dataset_dict[category_name])
            #print(f'train_data: {train_dataset_dict[category_name]}')
            #print(f'test data: {test_dataset_dict[category_name]}')
            model = Model
            optimizer = Optimizer
            model.to(Device)
            saving_param_path = weights_path+category_name+'.pt'
            # Print the initialized weights and biases for each layer
            #for name, param in model.named_parameters():
            #    print(f"{name}: {param}")

            training_and_evaluation(model, train_dataloader, test_dataloader, optimizer, category_name, parent_weights, loss_coef, path=saving_param_path)

hgru_models = hgru_model(son_parent_dict, train_dataset_dict, test_dataset_dict, categories_per_indent_dict, category_id_to_name_dict, weightspath)

categories_lists = list(categories_per_indent_dict.values())
categories_id = list(itertools.chain.from_iterable(categories_lists))
categories = []
for category_id in categories_id:
    categories.append(category_id_to_name_dict[category_id])
    

predictions_dict = get_results_on_test_set(weightspath, train_dataset_dict, test_dataset_dict, categories = categories)

with open(test_predictions_path, 'wb') as handle:
    pickle.dump(predictions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    categories_lists = list(categories_per_indent_dict.values())
categories_id = list(itertools.chain.from_iterable(categories_lists))
categories = []
for category_id in categories_id:
    categories.append(category_id_to_name_dict[category_id])

category_id_list = []
 
# list out keys and values separately
key_list = list(category_id_to_name_dict.keys())
val_list = list(category_id_to_name_dict.values())

for cat_name in categories:
    position = val_list.index(cat_name)
    category_id_list.append(key_list[position])
    

weights_dict = get_weights_per_category(category_id_list, '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/hgru/models_weights/', category_id_to_name_dict)

with open(hgru_model_weights_path, 'wb') as handle:
    pickle.dump(weights_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)