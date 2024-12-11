import sys
sys.path.insert(0, '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/')
from model.GRU_model import *
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim


SequenceLength = 12
Features = 1
OutputDim = 1
HiddenSize = 64
LayersDim = 1
DropoutProb = 0.0
Lr = 0.1
Criterion = nn.MSELoss()
Epochs = 100
BatchSize = 32
Year = 2020
loss_coef_1= 3.743092554196371*(10**(-12))
loss_coef_2= 4.015085187747426*(10**(-13))
alpha= 1.0
TbDirectory = 'tbs/'


train_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/data/train_dataset.pickle'
test_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/data/test_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_categories_per_indent_dict.pickle'
son_parent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_parent_dict.pickle'
weightspath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/models_weights/'
parent_to_son_list_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_parent_to_sons_list_dict.pickle'
hgru_model_weights_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_hgru_model_weights.pickle'
sgru_model_weights_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_sgru_model_weights.pickle'
coefficient_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/data/coefficient_dict.pickle'

# Loss Analysis params:
weightspath_1 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/models_weights_1/'
weightspath_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/models_weights_2/'
weightspath_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/models_weights_3/'
weightspath_1_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/models_weights_1_2/'
weightspath_1_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/models_weights_1_3/'
weightspath_2_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/models_weights_2_3/'



test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/test_predictions.pickle'
test_predictions_path_1 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/test_predictions_1.pickle'
test_predictions_path_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/test_predictions_2.pickle'
test_predictions_path_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/test_predictions_3.pickle'
test_predictions_path_1_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/test_predictions_1_2.pickle'
test_predictions_path_1_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/test_predictions_1_3.pickle'
test_predictions_path_2_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/bidirectional/test_predictions_2_3.pickle'


#Define our device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#Model=GRUModel(input_dim=Features, hidden_dim=HiddenSize, layer_dim=LayersDim, output_dim=OutputDim, dropout_prob=DropoutProb, seed=42)
Model=GRUWithAttentionModel(input_dim=Features, hidden_dim=HiddenSize, layer_dim=LayersDim, output_dim=OutputDim, dropout_prob=DropoutProb, seed=0)
Optimizer=torch.optim.AdamW(Model.parameters(), lr=Lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(Optimizer, mode='min', factor=0.9, patience=10, verbose=True)
