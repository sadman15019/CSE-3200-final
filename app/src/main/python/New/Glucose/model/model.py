
##--------------------------------------------------------------------------------
import torch
import numpy as np
from .config import config
#Seed
torch.manual_seed(config.SEED) 
np.random.seed(config.SEED)

def torch_model(inp):
    '''
        *  torch model with 1 input layer + 2 hidden layer + 1 output layer.
    '''
    n_hidden = 700
    model= torch.nn.Sequential(
        torch.nn.Linear(inp, n_hidden), ## input feature 48
        torch.nn.Dropout(0.5),  
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden, n_hidden),
        torch.nn.Dropout(0.5), 
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden, 1),
    )
    return model

##--------------------------------------------------------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers

def DNN(X):
    model = Sequential()

    # The Input Layer :
    model.add(Dense(100, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
    
    # The Hidden Layers :
    model.add(Dense(150, kernel_initializer='normal',activation='relu'))
    model.add(Dense(200, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250, kernel_initializer='normal',activation='relu'))
    model.add(Dense(300, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    
    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    
    # Compile the network :
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MAE'])
    
    return model

def DNN_3Layers(X):
    model = Sequential()

    # The Input Layer :
    model.add(Dense(100, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
    
    # The Hidden Layers :
    model.add(Dense(150, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, kernel_initializer='normal',activation='relu'))
    
    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    
    # Compile the network :
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MAE'])
    
    return model