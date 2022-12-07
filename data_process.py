import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

base_path = 'PATH TO TRAIN FOLDER THAT HAS THE 5 SNAKE IMAGE SUB FOLDERS'
Snake = np.array(['class-1', 'class-2', 'class-3', 'class-4', 'class-5'])


def construct_SNAKES(snake_path, iteration):
    #Here we import the path to the folder that contains the snake images
    #and then proceed to generate the arrays that will contain all their pixel
    #data. At the same time we also generate the arrays that will contain the
    #labels for the pictures, and return both the snake data array and the 
    #label array.
    
    path2imgs = os.path.join(base_path, snake_path)
    imgFileList = glob.glob(os.path.join(path2imgs,'*.jpg'))
    data = np.empty((len(imgFileList),256,256,3), dtype=np.uint8)
    for i, FileName in enumerate(imgFileList):
        img = cv2.imread(FileName, cv2.IMREAD_UNCHANGED)
        imgre = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        data[i, ...] = imgre
    return data

def save_SNAKES():
    #Here we itterate over the five snake folders and save the arrays into 
    #variables to then contruct a single array for all the snake pixel data 
    #and one for all the labels. We then split into training and test data and
    #save the arrays ad .npy files to be able to load them easily in the future
    #and not have to deal with loading the images every time we run the code.
    
    x1 = construct_SNAKES(Snake[0], 0)
    x2 = construct_SNAKES(Snake[1], 1)
    x3 = construct_SNAKES(Snake[2], 2)
    x4 = construct_SNAKES(Snake[3], 3)
    x5 = construct_SNAKES(Snake[4], 4)
    nsamples = len(x1)+len(x2)+len(x3)+len(x4)+len(x5)
    
    X = np.append(np.append(np.append(np.append(x1,x2, axis=0),x3, axis=0),x4,axis=0),x5,axis=0)

    Y = np.zeros((nsamples,5))    
    Y[:len(x1),0]=1
    Y[len(x1):(len(x1)+len(x2)),1]=1
    Y[(len(x1)+len(x2)):(len(x1)+len(x2)+len(x3)),2]=1
    Y[(len(x1)+len(x2)+len(x3)):(len(x1)+len(x2)+len(x3)+len(x4)),3]=1
    Y[(len(x1)+len(x2)+len(x3)+len(x4)):(len(x1)+len(x2)+len(x3)+len(x4)+len(x5)),4]=1
    
    del x1, x2, x3, x4, x5
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
    
    X_train = np.divide(X_train, 255, dtype='float16')
    X_test = np.divide(X_test, 255, dtype='float16')
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=.25, random_state=123
    )
    X_train = X_train.reshape((X_train.shape[0],196608)) #196,608 = 256*256*3
    X_val = X_val.reshape((X_val.shape[0],196608))
    X_test = X_test.reshape((X_test.shape[0],196608))
    
    np.save("snakes/X_train_full.npy", X_train)
    np.save("snakes/y_train_full.npy", y_train)
    np.save("snakes/X_test_full.npy", X_test)
    np.save("snakes/y_test_full.npy", y_test)
    np.save("snakes/X_val_full.npy", X_val)
    np.save("snakes/y_val_full.npy", y_val)
    
    
    
    
def get_Snakes():
    #Here we load the .npyu files, make another split for the validation data,
    #and then assign all our variables to a dictionary to easily fetch them 
    #when we wish to train our neural network.
    
    X_train = np.load("snakes/x_train_full.npy")
    y_train = np.load("snakes/y_train_full.npy")
    y_test = np.load("snakes/y_test_full.npy")
    X_test = np.load("snakes/x_test_full.npy")
    X_val = np.load("snakes/x_val_full.npy")
    y_val = np.load("snakes/y_val_full.npy")

    
    
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    return data

#If we run the code, it will execute the save_SNAKES function and make the 
#.npy files. 
if __name__ == '__main__':   
    save_SNAKES()
     