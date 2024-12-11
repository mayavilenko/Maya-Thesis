import torch
import numpy as np
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from pipeline_config import *

def create_dataloader(category_train_df, category_test_df):
    x_train = category_train_df.iloc[:,:-1].to_numpy()
    y_train = category_train_df.iloc[:,-1:].to_numpy()
    x_test = category_test_df.iloc[:,:-1].to_numpy()
    y_test= category_test_df.iloc[:,-1:].to_numpy()

    x_train= torch.from_numpy(x_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    x_test = torch.from_numpy(x_test).to(torch.float32)
    y_test = torch.from_numpy(y_test).to(torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader =  DataLoader(train_dataset, batch_size=BatchSize, shuffle=False)
    test_dataloader =  DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)
    return train_dataloader, test_dataloader


def custom_loss_optuna(category, category_indent, category_weights, son_parent_dict, y_true, y_pred, weights_dict, coefficient_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha):
    """
    y_true: The target tensor
    y_pred: The prediction tensor
    weights_dict: SGRU weight of every category
    loss_coef_1: regularization parameter for top-down addition
    loss_coef_2: regularization parameter for bottom-up addition
    loss_coef_3: regularization parameter for correlation_sign
    """
    mse = Criterion(y_pred, y_true)

    parent = son_parent_dict[category]
    if parent in list(weights_dict.keys()):
        parent_weights = weights_dict[parent]
        norm1 = torch.sum((category_weights - parent_weights)**2)
    else:
        norm1 = 0

    if category in list(parent_to_son_list_dict.keys()):
        sons = parent_to_son_list_dict[category]['sons']
        norm2_terms = []
        coefficient_lst = []
        for son in sons:
            son_weights = weights_dict[son]
            son_name = category_id_to_name_dict[son]
            coefficient = coefficient_dict[son_name]
            coefficient_lst.append(coefficient)
            norm2_term = torch.mul(coefficient, torch.sum((category_weights - son_weights)**2))
            norm2_terms.append(norm2_term)
        norm2_sum = sum(norm2_terms) ##weighted sum of son-category - vector multiplication
        norm2 = torch.sum(norm2_sum) #convert to torch
        coefficient_lst_sum = sum(coefficient_lst)
        norm2 = norm2/coefficient_lst_sum
    else:
        norm2 = 0
    
    # If the loss coeff is high we want to force the son model's weights to be as close to the parent's as possible.
    if loss_coef_1==0:
        if loss_coef_2 == 0:
            loss = mse   #since we'd like to maximise the sign, we subtract it from loss term
        else:
            loss = mse + loss_coef_2*norm2 #since we'd like to maximise the sign, we subtract it from loss term
    else:
        if loss_coef_2 == 0:
            loss = mse + loss_coef_1*norm1  #since we'd like to maximise the sign, we subtract it from loss term
        else:
            loss = mse + loss_coef_1*norm1 + loss_coef_2*norm2 

    loss = loss*np.exp(-alpha*category_indent) #we would like the loss to be larger for upper (smaller) indents so the model will focus more on these categories when hyperparameter tuning

    return loss


def custom_loss(category, category_indent, category_weights, son_parent_dict, y_true, y_pred, weights_dict, coefficient_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha):
    """
    y_true: The target tensor
    y_pred: The prediction tensor
    weights_dict: SGRU weight of every category
    loss_coef_1: regularization parameter for top-down addition
    loss_coef_2: regularization parameter for bottom-up addition
    loss_coef_3: regularization parameter for correlation_sign
    """
    mse = Criterion(y_pred, y_true)

    parent = son_parent_dict[category]
    if parent in list(weights_dict.keys()):
        parent_weights = weights_dict[parent]
        norm1 = torch.sum((category_weights - parent_weights)**2)
    else:
        norm1 = 0

    if category in list(parent_to_son_list_dict.keys()):
        sons = parent_to_son_list_dict[category]['sons']
        norm2_terms = []
        coefficient_lst = []
        for son in sons:
            son_weights = weights_dict[son]
            son_name = category_id_to_name_dict[son]
            coefficient = coefficient_dict[son_name]
            coefficient_lst.append(coefficient)
            norm2_term = torch.mul(coefficient, torch.sum((category_weights - son_weights)**2))
            norm2_terms.append(norm2_term)
        norm2_sum = sum(norm2_terms) ##weighted sum of son-category - vector multiplication
        norm2 = torch.sum(norm2_sum) #convert to torch
        coefficient_lst_sum = sum(coefficient_lst)
        norm2 = norm2/coefficient_lst_sum
    else:
        norm2 = 0
    
    # If the loss coeff is high we want to force the son model's weights to be as close to the parent's as possible.
    if loss_coef_1==0:
        if loss_coef_2 == 0:
            loss = mse   #since we'd like to maximise the sign, we subtract it from loss term
        else:
            loss = mse + loss_coef_2*norm2 #since we'd like to maximise the sign, we subtract it from loss term
    else:
        if loss_coef_2 == 0:
            loss = mse + loss_coef_1*norm1  #since we'd like to maximise the sign, we subtract it from loss term
        else:
            loss = mse + loss_coef_1*norm1 + loss_coef_2*norm2 

    return loss


def unify_model_weights(model):
    param_dict ={}
    for name, param in model.named_parameters():
        param_dict[name] = param

    param_dict['gru.bias_hh_l0'] = param_dict['gru.bias_hh_l0'].view(-1,1)
    param_dict['gru.bias_ih_l0'] = param_dict['gru.bias_ih_l0'].view(-1,1)
    unified_weights = torch.hstack((
            param_dict['gru.weight_ih_l0'],
            param_dict['gru.weight_hh_l0'],
            param_dict['gru.bias_ih_l0'],
            param_dict['gru.bias_hh_l0']))

    return unified_weights

def training_loop_optuna(model, category_indent, train_dataloader, optimizer, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha):
    running_loss = 0
    running_mse_loss=0

    model.train()
    for inputs, labels in train_dataloader:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        inputs, labels = inputs.to(Device), labels.to(Device)
        #Changing input shape - last batch size can change so we define it as input.shape[0]
        inputs = inputs.view(inputs.shape[0], SequenceLength, Features) 
        #model prediction
        pred = model(inputs)
        # calculate loss
        category_weights = unify_model_weights(model)  
        #print(f'inputs: {inputs}')
        loss = custom_loss_optuna(category, category_indent, category_weights, son_parent_dict, labels, pred, weights_dict, coefficient_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)
        mse_loss = Criterion(pred, labels) #removed .view(-1,1) from labels
        # calculate the gradient given the custom loss function
        loss.backward()
        # update parameters
        optimizer.step()
        #Add to loss of batch to epoch train loss
        running_loss+=loss.item()
        # for tensor board
        running_mse_loss += mse_loss.item()
    # Calculte the epoch train loss
    epoch_train_loss = running_loss/len(train_dataloader.dataset)
    epoch_train_mse_loss = running_mse_loss/len(train_dataloader.dataset)

    return epoch_train_loss, epoch_train_mse_loss

def training_loop(model, category_indent, train_dataloader, optimizer, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha):
    running_loss = 0
    running_mse_loss=0

    model.train()
    for inputs, labels in train_dataloader:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        inputs, labels = inputs.to(Device), labels.to(Device)
        #Changing input shape - last batch size can change so we define it as input.shape[0]
        inputs = inputs.view(inputs.shape[0], SequenceLength, Features) 
        #model prediction
        pred = model(inputs)
        # calculate loss
        category_weights = unify_model_weights(model)  
        #print(f'inputs: {inputs}')
        loss = custom_loss(category, category_indent, category_weights, son_parent_dict, labels, pred, weights_dict, coefficient_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)
        mse_loss = Criterion(pred, labels) #removed .view(-1,1) from labels
        # calculate the gradient given the custom loss function
        loss.backward()
        # update parameters
        optimizer.step()
        #Add to loss of batch to epoch train loss
        running_loss+=loss.item()
        # for tensor board
        running_mse_loss += mse_loss.item()
    # Calculte the epoch train loss
    epoch_train_loss = running_loss/len(train_dataloader.dataset)
    epoch_train_mse_loss = running_mse_loss/len(train_dataloader.dataset)

    return epoch_train_loss, epoch_train_mse_loss

def evaluation_loop_optuna(model, category_indent, test_dataloader, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha):
    # Evaluation
    # Initiate test loss and mse test loss to zero
    test_loss = 0
    test_mse_loss = 0

    # Change model to eval mode
    model.eval()
    category_weights = unify_model_weights(model)
    # we dont need to update weights, so we define no_grad() to save memory
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            #print(f'last value 2 shape: {last_value.shape}')
            inputs = inputs.view(inputs.shape[0], SequenceLength, Features)
            inputs, labels = inputs.to(Device), labels.to(Device)
            #print(f'input shape: {inputs.shape}')
            out = model(inputs)
            test_batch_loss = custom_loss_optuna(category, category_indent, category_weights, son_parent_dict, labels, out, weights_dict, coefficient_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)
            test_batch_mse_loss = Criterion(out, labels)
            test_loss += test_batch_loss.item()
            test_mse_loss += test_batch_mse_loss.item()

        # Calculate epoch loss
    epoch_test_loss = test_loss/len(test_dataloader.dataset)
    epoch_mse_test_loss = test_mse_loss/len(test_dataloader.dataset)

    return epoch_test_loss, epoch_mse_test_loss

def evaluation_loop(model, category_indent, test_dataloader, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha):
    # Evaluation
    # Initiate test loss and mse test loss to zero
    test_loss = 0
    test_mse_loss = 0

    # Change model to eval mode
    model.eval()
    category_weights = unify_model_weights(model)
    # we dont need to update weights, so we define no_grad() to save memory
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            #print(f'last value 2 shape: {last_value.shape}')
            inputs = inputs.view(inputs.shape[0], SequenceLength, Features)
            inputs, labels = inputs.to(Device), labels.to(Device)
            #print(f'input shape: {inputs.shape}')
            out = model(inputs)
            test_batch_loss = custom_loss(category, category_indent, category_weights, son_parent_dict, labels, out, weights_dict, coefficient_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)
            test_batch_mse_loss = Criterion(out, labels)
            test_loss += test_batch_loss.item()
            test_mse_loss += test_batch_mse_loss.item()

        # Calculate epoch loss
    epoch_test_loss = test_loss/len(test_dataloader.dataset)
    epoch_mse_test_loss = test_mse_loss/len(test_dataloader.dataset)

    return epoch_test_loss, epoch_mse_test_loss

def training_and_evaluation_optuna(trial, model, category_indent, train_dataloader, test_dataloader, optim, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha, path): 
    #Create writer for using tesndorboard
    #category_name = category_id_to_name_dict[category]
    #writer = SummaryWriter(log_dir=f'{TbDirectory}_{category_name}')

    min_test_loss = np.inf

    for epoch in range(Epochs):
        epoch_train_loss, epoch_train_mse_loss = training_loop_optuna(model, category_indent, train_dataloader, optim, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)
        epoch_test_loss, epoch_test_mse_loss= evaluation_loop_optuna(model, category_indent, test_dataloader, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)

        checkpoint = {
         'epoch': epoch + 1,
         'valid_loss_min': epoch_test_loss,
         'state_dict': model.state_dict(),
         'optimizer': optim.state_dict(),
        }

        if epoch_test_loss <= min_test_loss:
            save_checkpoint(checkpoint, path)
            min_test_loss = epoch_test_loss

        #print(f'Category: {category}, epoch_test_loss: {min_test_loss}')

        ## hyperparameter tuning using optuna
        trial.report(min_test_loss, epoch)

    # hyperparameter tuning using optuna
    return min_test_loss

def training_and_evaluation(model, category_indent, train_dataloader, test_dataloader, optim, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha, path): 
    #Create writer for using tesndorboard
    #category_name = category_id_to_name_dict[category]
    #writer = SummaryWriter(log_dir=f'{TbDirectory}_{category_name}')

    min_test_loss = np.inf

    for epoch in range(Epochs):
        epoch_train_loss, epoch_train_mse_loss = training_loop(model, category_indent, train_dataloader, optim, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)
        epoch_test_loss, epoch_test_mse_loss= evaluation_loop(model, category_indent, test_dataloader, category, weights_dict, coefficient_dict, son_parent_dict, parent_to_son_list_dict, category_id_to_name_dict, loss_coef_1, loss_coef_2, alpha)

        checkpoint = {
         'epoch': epoch + 1,
         'valid_loss_min': epoch_test_loss,
         'state_dict': model.state_dict(),
         'optimizer': optim.state_dict(),
        }

        if epoch_test_loss <= min_test_loss:
            save_checkpoint(checkpoint, path)
            min_test_loss = epoch_test_loss

        #print(f'Category: {category}, epoch_test_loss: {min_test_loss}')

    # hyperparameter tuning using optuna
    return min_test_loss


def save_checkpoint(checkpoint, path):
    """
    checkpoint: checkpoint we want to save
    path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(checkpoint, path)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min

def get_predictions_on_test_set(model, test_dataloader):
    # Evaluation
    # Change model to eval mode
    model.eval()
    predictions_list = []
    # we dont need to update weights, so we define no_grad() to save memory
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.view(inputs.shape[0], SequenceLength, Features)
            inputs, labels = inputs.to(Device), labels.to(Device)
            out = model(inputs)
            predictions_list.append(out.view(1,-1))
    # Calculate epoch loss
    epoch_predictions = torch.cat(predictions_list, dim=1)
    return epoch_predictions
