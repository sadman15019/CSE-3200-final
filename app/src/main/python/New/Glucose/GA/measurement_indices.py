#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    @author: Md. Rezwanul Haque
'''
#-------------------------
# imports
#-------------------------
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from .config import config
import sys
sys.path.append('../')
from PPG.utils import create_dir
np.random.seed(config.SEED)
plt.style.use('classic')

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearsonr(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def rmse(targets,predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def filter_nan(s,o):
    """
    this functions removed the data  from simulated and observed data
    whereever the observed data contains nan
    
    this is used by all other functions, otherwise they will produce nan as 
    output
    """
    if np.sum(~np.isnan(s*o))>=1:
        data = np.array([s.flatten(),o.flatten()])
        data = np.transpose(data)
        data = data[~np.isnan(data).any(1)]
        s = data[:,0]
        o = data[:,1]
    return s, o

def index_agreement(s, o):
    """
	index of agreement
	
	Willmott (1981, 1982) 
	input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    s,o = filter_nan(s,o)
    ia = 1 -(np.sum((o-s)**2))/(np.sum(
    			(np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    return ia

def bland_altman_plot_paper(data1, data2, name = " ", *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    print(md + 1.96*sd, md - 1.96*sd)
    
    fig, ax = plt.subplots(figsize=(5,3))
    fig.set_facecolor("white")
    # plt.title('Bland-Altman Plot')
#    plt.legend()
    plt.scatter(mean, diff, *args, **kwargs)

    plt.axhline(md + 1.96*sd, color='g', label="md + 1.96*sd", linestyle='--')
    plt.axhline(md,           color='r', label="md",           linestyle='--')
    plt.axhline(md - 1.96*sd, color='b', label="md - 1.96*sd", linestyle='--')

    labels = ["md + 1.96*sd", "md", "md - 1.96*sd"]
    handles, _ = ax.get_legend_handles_labels()

    # Slice list to remove first handle
    plt.legend(handles = handles[:], labels = labels)
    plt.xlabel("Average Glucose(mmol/L)",fontsize=10)
    plt.ylabel("Difference Glucose(mmol/L)",fontsize=10)
    plt.savefig("H:/Glucose/output_figs/"+ str(name)+".pdf", bbox_inches='tight', dpi=320)
    plt.show()
  
def act_pred_plot_paper(y, predicted, R_2=None, mae=None, name = ""):
    fig, ax = plt.subplots(figsize=(5,3))
    fig.set_facecolor("white")
    ax.text(y.min()-3, y.max()-1, '$MAE =$ %0.3f' %(np.mean(mae)))
    ax.text(y.min()-3, y.max()+1, '$R^2 =$ %.3f' %(np.mean(R_2)))
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=1)
    ax.set_xlabel('Reference Glucose(mmol/L)')
    ax.set_ylabel('Estimated Glucose(mmol/L)')
    plt.savefig("H:/Glucose/output_figs/"+ str(name)+".pdf", bbox_inches='tight', dpi=320)   
    plt.show() 
    
##=========================Bland-Altman plot 
def bland_altman_plot(data1, data2, label_name, save_dir, GA, verbose=1, *args, **kwargs):
    if GA == "With-GA":
        data1     = np.asarray(data1)
        data2     = np.asarray(data2)
        mean      = np.mean([data1, data2], axis=0)
        diff      = data1 - data2                   # Difference between data1 and data2
        md        = np.mean(diff)                   # Mean of the difference
        sd        = np.std(diff, axis=0)            # Standard deviation of the difference

        fig, ax = plt.subplots()
        plt.title('Bland-Altman Plot')
        # plt.legend()
        plt.scatter(mean, diff, *args, **kwargs)

        plt.axhline(md + 1.96*sd, color='g', label="md + 1.96*sd", linestyle='--')
        plt.axhline(md,           color='r', label="md",           linestyle='--')
        plt.axhline(md - 1.96*sd, color='b', label="md - 1.96*sd", linestyle='--')

        labels = ["md + 1.96*sd", "md", "md - 1.96*sd"]
        handles, _ = ax.get_legend_handles_labels()

        # Slice list to remove first handle
        plt.legend(handles = handles[:], labels = labels)
        x_label = "Average "+label_name
        y_label = "Difference "+label_name
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        _path = create_dir(save_dir, "Fig_GA")
        # _img_save_dir = os.path.join(save_dir, _path)
        plt.savefig(os.path.join(_path, label_name+"_Bland-Altman.png"), dpi = 100)
        if verbose:
            plt.show()
        
    else:
        data1     = np.asarray(data1)
        data2     = np.asarray(data2)
        mean      = np.mean([data1, data2], axis=0)
        diff      = data1 - data2                   # Difference between data1 and data2
        md        = np.mean(diff)                   # Mean of the difference
        sd        = np.std(diff, axis=0)            # Standard deviation of the difference

        fig, ax = plt.subplots()
        plt.title('Bland-Altman Plot')
        # plt.legend()
        plt.scatter(mean, diff, *args, **kwargs)

        plt.axhline(md + 1.96*sd, color='g', label="md + 1.96*sd", linestyle='--')
        plt.axhline(md,           color='r', label="md",           linestyle='--')
        plt.axhline(md - 1.96*sd, color='b', label="md - 1.96*sd", linestyle='--')

        labels = ["md + 1.96*sd", "md", "md - 1.96*sd"]
        handles, _ = ax.get_legend_handles_labels()

        # Slice list to remove first handle
        plt.legend(handles = handles[:], labels = labels)
        x_label = "Average "+label_name
        y_label = "Difference "+label_name
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        _path = create_dir(save_dir, "Fig_Without_GA")
        # _img_save_dir = os.path.join(save_dir, _path)
        plt.savefig(os.path.join(_path, label_name+"_Bland-Altman.png"), dpi = 100)
        if verbose:
            plt.show()

def act_pred_plot(y, predicted, label_name, save_dir, r=None, st=None, GA = None, verbose=1):
    if GA == "With-GA":
        fig, ax = plt.subplots()

        ax.text(y.min(), y.max(), str('\n\n$MAE$ = %0.3f$\pm%0.3f$\n' %(r[0],np.std(st[0])/10.0)))
        ax.text(y.min(), y.max()-1, str('$R^2$ = %.3f$\pm%0.3f$' %(r[1], np.std(st[1])/10.0)))

        ax.scatter(y, predicted)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=1)
        x_label = "Reference "+label_name
        y_label = "Estimated "+label_name
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        _path = create_dir(save_dir, "Fig_GA")
        # _img_save_dir = os.path.join(save_dir, _path)
        plt.savefig(os.path.join(_path, label_name+"_Act_vs_Pred.png"), dpi = 100)
        if verbose:
            plt.show()
        
    else:
        fig, ax = plt.subplots()

        ax.text(y.min(), y.max(), str('\n\n$MAE$ = %0.3f$\pm%0.3f$\n' %(r[0],np.std(st[0])/10.0)))
        ax.text(y.min(), y.max()-1, str('$R^2$ = %.3f$\pm%0.3f$' %(r[1], np.std(st[1])/10.0)))

        ax.scatter(y, predicted)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=1)
        x_label = "Reference "+label_name
        y_label = "Estimated "+label_name
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        _path = create_dir(save_dir, "Fig_Without_GA")
        # _img_save_dir = os.path.join(save_dir, _path)
        plt.savefig(os.path.join(_path, label_name+"_Act_vs_Pred.png"), dpi = 100)
        if verbose:
            plt.show()