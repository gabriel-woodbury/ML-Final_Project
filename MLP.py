# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:10:19 2022

@author: Gabriel Woodbury
"""

import numpy as np
import copy
import time
from sklearn.utils import shuffle


np.random.seed(4)
class MLP:
    
    def __init__(self):
        
        self.epochs = 1
        self.HiddenLayer = []
        self.w = []
        self.b = []
        self.phi = []
        self.mu = []
        self.eta = 1
        self.momentum = 0.1
        self.reg = 1
        self.W_momentum = []
        self.act_dic = {'LReLU': LReLU_act, 'ReLU': ReLU_act, 'sigmoid': sigmoid_act}
    
    #The method we will use to add layers to the Neural Network
    
    def add(self, nodes: int, activation: str ):
        self.HiddenLayer.append((nodes, activation))
        
    #Here we have our feed forward method, which is evaluating the activation
    #function at the dot produc of the data with the weights vector plus a bias
        
    @staticmethod
    def FeedForward(w, b, phi, x):
        return phi(np.dot(w, x) + b)
            
    '''
    BackPropagation algorithm implementing the Gradient Descent 
    '''
  
    def BackPropagation(self, x, z, Y, w, b, phi):
        #z are the outputs of each layer, phi are the activation functions
        self.delta = []
        
        
        # We initialize a w and b that are used only inside the backpropagation algorithm       
        self.W = []
        self.B = []
        
        # We start computing the LAST error, the one for the OutPut Layer 
        self.delta.append(  (z[-1] - Y) * phi[-1](z[-1], der=True) )
        
        '''Now we BACKpropagate'''
        # We thus compute from next-to-last to first
        for i in range(0, len(z)-1):
            self.delta.append( np.dot( self.delta[i], w[- 1 - i] ) * phi[- 2 - i](z[- 2 - i], der=True) )
        
        # We have the error array ordered from last to first; we flip it to order it from first to last
        self.delta = np.flip(self.delta, 0)  
        
        '''GRADIENT DESCENT'''
        
        
        #First itteration does not have momentum hence we do not include it in first calculation

            
        if self.first_run==1:
            
            
         
            self.W_momentum.append( - self.grad_desc_const * w[0]- self.eta * np.kron(self.delta[0], x).reshape( len(z[0]), x.shape[0] ) )
            
    
            # We start from the first layer that is special, since it is connected to the Input Layer
            self.W.append( w[0] + self.W_momentum[0]  )
            self.B.append( b[0] - self.eta * self.delta[0] )
            
            
            
            
            # We now descend for all the other Hidden Layers + OutPut Layer
            for i in range(1, len(z)):
                self.W_momentum.append( - self.grad_desc_const * w[i] - self.eta * np.kron(self.delta[i], z[i-1]).reshape(len(z[i]), len(z[i-1])))
                
                self.W.append( w[i] + self.W_momentum[i] )
                self.B.append( b[i] - self.eta * self.delta[i] )
            
            # We return the descended parameters w, b
            return self.W ,self.B
    
                
        
        else:
            self.W_momentum2 = []
            
            self.W_momentum2.append(-self.grad_desc_const * w[0] - self.eta * np.kron(self.delta[0], x).reshape( len(z[0]), x.shape[0] )  )
            self.W_momentum[0] = copy.deepcopy( self.W_momentum2[0] + self.momentum * self.W_momentum[0] )
            
            # We start from the first layer that is special, since it is connected to the Input Layer
            self.W.append( w[0] + self.W_momentum[0] )
            self.B.append( b[0] - self.eta * self.delta[0] )
            
            # We now descend for all the other Hidden Layers + OutPut Layer
            for i in range(1, len(z)):
                
                self.W_momentum2.append(-self.grad_desc_const * w[i] - self.eta * np.kron(self.delta[i], z[i-1]).reshape(len(z[i]), len(z[i-1]))  )
                self.W_momentum[i] = copy.deepcopy(self.W_momentum2[i] + self.momentum * self.W_momentum[i] )
                
                self.W.append( w[i] + self.W_momentum[i] )
                self.B.append( b[i] - self.eta * self.delta[i] )
            
            
            
            # We return the descended parameters w, b
            return self.W, self.B
        
    
    '''
    Fit method: it calls FeedForward and Backpropagation methods
    '''
    def Fit(self, X_train, Y_train):            
        print('Start fitting...')
        '''
        Input layer
        '''
        self.X = X_train
        self.Y = Y_train
        
        '''
        We now initialize the Network by retrieving the Hidden Layers and concatenating them 
        '''
        print('Model recap: \n')
        print('You are fitting an ANN with the following amount of layers: ', len(self.HiddenLayer))
        
        for i in range(0, len(self.HiddenLayer)) :
            print('Layer ', i+1)
            print('Number of neurons: ', self.HiddenLayer[i][0])
            if i==0:
                # We now try to use the He et al. Initialization from ArXiv:1502.01852
                self.w.append( np.random.randn(self.HiddenLayer[i][0] , self.X.shape[1])/np.sqrt(2/self.X.shape[1]) )
                self.b.append( np.random.randn(self.HiddenLayer[i][0])/np.sqrt(2/self.X.shape[1]))
            
            else :
                # We now try to use the He et al. Initialization from ArXiv:1502.01852
                self.w.append( np.random.randn(self.HiddenLayer[i][0] , self.HiddenLayer[i-1][0] )/np.sqrt(2/self.HiddenLayer[i-1][0]))
                self.b.append( np.random.randn(self.HiddenLayer[i][0])/np.sqrt(2/self.HiddenLayer[i-1][0]))
         
               # Initialize the Activation function
            
            self.phi.append(self.act_dic[self.HiddenLayer[i][1]])
            print('\tActivation: ', self.HiddenLayer[i][1])
            
        '''
        Now we start the Loop over the training dataset
        '''  
        self.grad_desc_const = (self.eta * self.reg) 
        for epoch in range(self.epochs):
            self.X, self.Y = shuffle(self.X, self.Y)
            self.current_epoch = epoch + 1
            print("Starting Backpropagation")
            s_time = time.time()
            for I in range(0, self.X.shape[0]): # loop over the training set
                '''
                Now we start the feed forward
                '''  
                self.first_run = I+1
                self.z = []
                
                self.z.append( self.FeedForward(self.w[0], self.b[0], self.phi[0], self.X[I]) ) # First layers
                
                for i in range(1, len(self.HiddenLayer)): #Looping over layers
                    self.z.append( self.FeedForward(self.w[i] , self.b[i], self.phi[i], self.z[i-1] ) )
            
                
                '''
                Here we backpropagate
                '''      
                self.w, self.b  = self.BackPropagation(self.X[I], self.z, self.Y[I], self.w, self.b, self.phi)
                
                '''
                Compute cost function
                ''' 
                regsum = 0
                for ind in range(len(self.w)):
                     regsum += np.sum(np.square(self.w[ind]))
                
                calc_error = (1/2) * ( np.dot(self.z[-1] - self.Y[I], self.z[-1] - self.Y[I]) )  + self.reg * regsum
                self.mu.append(calc_error)
                if calc_error < 1.0:
                    break
                
                
            e_time = time.time()
            t_time = e_time - s_time
            print('Fit done for epoch', self.current_epoch)
            print("Time for current epoch,", t_time )
            print('Error at this epoch', self.mu[-1], '. \n')
        
    '''
    predict method
    '''
    def predict(self, X_test):
        
        print('Starting predictions...')
        
        self.pred = []
        
        for I in range(0, X_test.shape[0]): # loop over the training set
            
            '''
            Now we start the feed forward
            '''  
            self.z = []
            
            self.z.append(self.FeedForward(self.w[0] , self.b[0], self.phi[0], X_test[I])) #First layer
    
            for i in range(1, len(self.HiddenLayer)) : # loop over the layers
                self.z.append( self.FeedForward(self.w[i] , self.b[i], self.phi[i], self.z[i-1]))
            
          
            # Append the prediction;
            self.pred.append( np.argmax(self.z[-1])) # NB: self.z[-1]  is the last element of the self.z list
        
        print('Predictions done. \n')

        return np.array(self.pred)
      
    
    #Method to return accuracy values
    
    def get_accuracy(self):
        return np.array(self.mu)
    
    #Method to set the learning rate

    def set_learning_rate(self, et=1):
        self.eta = et 
        
    #Method to set the momentum
    
    def set_momentum(self, mom=0.1):
        self.momentum = mom
        
    
    #Method to set the Regularization Constant
    
    def set_reg_const(self, reg=1):
        self.reg = reg
     
    
    #Method to set the number of epochs
    
    def set_epochs(self, epoch=1):
        self.epochs = epoch
    

#Here we will define the Sigmoid activation function
    
def sigmoid_act(x, der=False):
    sigmoid = 1 / (1 + np.exp(- x))
    
    if (der==True):
        sig = sigmoid * (1 - sigmoid)
        return sig
    
    else:
        return sigmoid
    
#Here we will define the ReLU activation function

def ReLU_act(x, der=False):
    
    if (der == True): # the derivative of the ReLU is the Heaviside Theta
        ReLU = np.heaviside(x, 1)
        
    else :
        ReLU = np.maximum(x, 0)
        
    return ReLU

#Here we will define the Leaky ReLU activator

def LReLU_act(x, der=False):
    
    if (der == True):
        LReLU = np.heaviside(x,1) + 0.01*np.heaviside(-x,0)
    else :
        LReLU = np.maximum(x,0) + 0.01*np.minimum(x,0)
    return LReLU
    
    
    
    
    
        
                