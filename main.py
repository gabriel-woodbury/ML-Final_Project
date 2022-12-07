# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:31:11 2022

@author: Gabriel Woodbury
"""

from MLP import MLP
import numpy as np
from data_process import get_Snakes


def label_maker(label_list):
    labels = []
    for ind in range(len(label_list)):
        labels.append(np.argmax(label_list[ind]))
    return labels

def accuracy(pred_label, label):
    accuracy_list = []
    for ind in range(len(label)):
        accuracy_list.append((pred_label[ind]==label[ind])*1)
    acc = np.mean(accuracy_list)
    return acc

if __name__ == '__main__':   
    #For get snakes, use any number between 1 and 4
    data = get_Snakes()
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    X_val, y_val = data['X_val'], data['y_val']
    
    del data
    
    model = MLP()
    

    model.add(1, 'sigmoid')
    model.add(5, 'sigmoid')

    model.set_learning_rate(0.0001)
    model.set_momentum(0.4)
    model.set_reg_const(10000)
    model.set_epochs(1)
    
    model.Fit(X_train, y_train)
    acc_val = model.get_accuracy()
    
    predictions_test = model.predict(X_test)
    predictions_val = model.predict(X_val)
    
    labels_test = label_maker(y_test)
    labels_val = label_maker(y_val)
    
    acc_test = accuracy(predictions_test, labels_test)
    acc_vald = accuracy(predictions_val, labels_val)
