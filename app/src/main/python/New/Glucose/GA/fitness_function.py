#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    @author: Md. Rezwanul Haque
'''
#-------------------------
# imports
#-------------------------
import math 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, r2_score
import torch
from torch.autograd import Variable
from sklearn import metrics
from .config import config
from .measurement_indices import pearsonr
np.random.seed(config.SEED)

class FitnessFunction:
    def __init__(self, n_splits=5, *args, **kwargs):
        """
        Parameters
        -----------
        n_splits :int,
            Number of splits for cv
        verbose: 0 or 1
        """
        self.n_splits = n_splits

    def calculate_fitness_wrapper(self, model, x, y, flag):
        '''
            * Calculate fitness function with pearson's R of kfold model
                @args:
                    x     :  Features
                    y     :  label values
        '''

        if flag == "torch":
            cv_set = np.repeat(-1.,x.shape[0])
            
            cv = KFold(n_splits=config.N_SPLITS, random_state=config.RANDOM_STATE, shuffle=True)

            model = model(x.shape[1])
            
            for train_index, test_index in cv.split(x):
                # print("Train Index: ", train_index, "\n")
                # print("Test Index: ", test_index)
            
                X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
                
                if X_train.shape[0] != y_train.shape[0]:
                    raise Exception()
                    
                y_train = y_train.to_numpy()
                y_test = y_test.to_numpy()
                
                X_train = Variable(torch.from_numpy(X_train))
                X_test = Variable(torch.from_numpy(X_test))
                
                y_train = Variable(torch.from_numpy(y_train))
                y_test = Variable(torch.from_numpy(y_test))
            
                    
                criterion = torch.nn.MSELoss(size_average = False)
                optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

                ###### Epoch: 500
                epochs = config.TORCH_EPOCHS
            
                for epoch in range(epochs):
                    #make a prediction
                    yhat_drop = model(X_train.float())
                    
                    #calculate the loss
                    loss_drop = criterion(yhat_drop, y_train.float())
                    
                    #clear gradient 
                    optimizer.zero_grad()
                    
                    #Backward pass: compute gradient of the loss with respect to all the learnable parameters
                    loss_drop.backward()
                    
                    #the step function on an Optimizer makes an update to its parameters
                    optimizer.step()
                ######################################
                
                
                ## Cal. pred for test set
                    
                model.eval()
                yhat_drop_pred = model(X_test.float())
                
                # print("Individual R^2 Score: " + str(metrics.r2_score(y_test.numpy(), yhat_drop_pred.detach().numpy())))
                # cv_set[test_index] = yhat_drop_pred.detach().numpy().reshape(len(yhat_drop_pred),)
                
                return metrics.r2_score(y_test.numpy(), yhat_drop_pred.detach().numpy())
        else: 
            cv_set = np.repeat(-1.,x.shape[0])
            # skf = StratifiedKFold(n_splits=self.n_splits)
            skf = KFold(n_splits = self.n_splits ,shuffle=True, random_state=42)
            ## Import model
            model = model
            for train_index,test_index in skf.split(x,y):
                x_train,x_test = x[train_index],x[test_index]
                y_train,y_test = y[train_index],y[test_index]
                if x_train.shape[0] != y_train.shape[0]:
                    raise Exception()
                model.fit(x_train,y_train)
                predicted_y = model.predict(x_test)
                cv_set[test_index] = predicted_y
            return pearsonr(y,cv_set)
    
    def calculate_fitness_filter(self, X, y):
        '''
            * Calculate fitness function: correlation-based feature selection (CFS).
                @args:
                    X     :  Features
                    y     :  label values
                return:
                    R     :  calculate corelation-coefficient(R) with features.
        '''
        
        row = X.shape[0]
        col = X.shape[1]
        feat = col
        
        DX = []
        DY = []

        for i in range(row):
          for j in range(col):
            Dy = (y[i] - y[j])
            sm = 0
            if Dy >= 0:
              sm = sm
              for k in range(feat):
                # print(i, j, k)
                sm += (X[i][k] - X[j][k])**2
        
            elif Dy < 0:
              sm = -sm
              for k in range(feat):
                # print(i, j, k)
                sm += (X[i][k] - X[j][k])**2
            
            ## Sum
            DY.append(Dy)
            DX.append(math.sqrt(sm / feat))
              
        ###================ Calculate SDxDy
        DX_mean = np.mean(DX)
        # print(DX_mean)
        DY_mean = np.mean(DY)
        # print(DY_mean)
        
        sm_DX_DY = 0
        for i in range(len(DX)):
            sm_DX_DY += (DX[i] - DX_mean) * (DY[i] - DY_mean)
        
        SDxDy = sm_DX_DY / (feat-1)
        # print("SDxDy : " + str(SDxDy))
            
        ###===============Calculte SDx
        sm_DX = 0
        for i in range(len(DX)):
            sm_DX += (DX[i] - DX_mean)**2
            
        SDx = sm_DX / (feat-1)
        # print("SDx : " + str(SDx))
        
        ###========== Calculte SDy
        sm_Dy = 0
        for i in range(len(y)):
            sm_Dy += (DY[i] - DY_mean)**2
        
        SDy = sm_Dy / (feat - 1)
        # print("SDy : " + str(SDy))    
        
        #### Now Calculate corelation-Coefficient: R
        R = (SDxDy) / math.sqrt(SDx * SDy)
        # print("R : " +str(R)) 
        return R