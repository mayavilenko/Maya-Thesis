import torch
import numpy as np
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from pipeline_config import *

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

def create_dataloader(category_train_df, category_test_df):
    x_train = category_train_df.iloc[:,:-1].to_numpy()
    y_train = category_train_df.iloc[:,-1].to_numpy()
    x_test = category_test_df.iloc[:,:-1].to_numpy()
    y_test= category_test_df.iloc[:,-1].to_numpy()

    x_train= torch.from_numpy(x_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    x_test = torch.from_numpy(x_test).to(torch.float32)
    y_test = torch.from_numpy(y_test).to(torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader =  DataLoader(train_dataset, batch_size=BatchSize, shuffle=False)
    test_dataloader =  DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)
    return train_dataloader, test_dataloader

def custom_loss(y_true, y_pred, son_weights, parent_weights, loss_coef):
    """
    y_true: The target tensor
    y_pred: The prediction tensor
    son_weights: The son's model weights
    parent_weights: The parent's model weights
    loss_coef: A float between 0 and 1 calculated by multiplying the correlation between the parent and the son with alpha (an hyperparameter).
    """
    mse = Criterion(y_pred, y_true.view(-1,1))
    # If the loss coeff is high we want to force the son model's weights to be as close to the parent's as possible.
    norm = torch.sum((parent_weights - son_weights)**2)
    if loss_coef==0:
        loss = mse
    else:
        loss = loss_coef*norm + mse
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

def training_loop(model, train_dataloader, optimizer, parent_weights, loss_coef):
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
        son_weights = unify_model_weights(model)
        loss = custom_loss(labels, pred, son_weights, parent_weights, loss_coef)
        mse_loss = Criterion(pred, labels.view(-1,1))
        #loss = Criterion(pred, labels.view(-1,1))
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

def evaluation_loop(model, test_dataloader, parent_weights, loss_coef):
    # Evaluation
    # Initiate test loss and mse test loss to zero
    test_loss = 0
    test_mse_loss = 0
    # Change model to eval mode
    model.eval()
    son_weights = unify_model_weights(model)
    # we dont need to update weights, so we define no_grad() to save memory
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.view(inputs.shape[0], SequenceLength, Features)
            inputs, labels = inputs.to(Device), labels.to(Device)
            out = model(inputs)
            test_batch_loss = custom_loss(labels, out, son_weights, parent_weights, loss_coef)
            test_batch_mse_loss = Criterion(out, labels.view(-1,1))

            test_loss += test_batch_loss.item()
            test_mse_loss += test_batch_mse_loss.item()
    # Calculate epoch loss
    epoch_test_loss = test_loss/len(test_dataloader.dataset)
    epoch_mse_test_loss = test_mse_loss/len(test_dataloader.dataset)

    return epoch_test_loss, epoch_mse_test_loss

def training_and_evaluation(model, train_dataloader, test_dataloader, optim, category, parent_weights, loss_coef, path): #early_stopping=True):
    ##Create writer for using tesndorboard
    #writer = SummaryWriter(log_dir=f'{TbDirectory}_{category}')
    #current_loss=np.inf
    #patience_counter = 0

    min_test_loss = np.inf

    for epoch in range(Epochs):
        epoch_train_loss, epoch_train_mse_loss = training_loop(model, train_dataloader, optim, parent_weights, loss_coef)
        epoch_test_loss, epoch_test_mse_loss= evaluation_loop(model, test_dataloader, parent_weights, loss_coef)

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
        #trial.report(min_test_loss, epoch)

    # hyperparameter tuning using optuna
    #return min_test_loss

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
