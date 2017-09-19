# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:04:09 2017

@author: pairwin
Adding a comment for testing branch
"""
import itertools
import pyodbc
import logging
import pandas as pd
import sklearn.model_selection as cv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np



class Helper:
    def __init__(self):
        self.data = None
        self.databasecon = r'DRIVER={NetezzaSQL};SERVER=SRVDWHITP01;DATABASE=EDW_SPOKE;UID=pairwin;PWD=pairwin;TIMEOUT=0'        
        logging.basicConfig(level=logging.INFO)
        logging.info('Initializing...')
        
    def getSQL(self, path):
        with open(path, 'r') as myfile:
            self.data = myfile.read();
            return self.data
        
    def readData(self, sql):
        '''
            Returns Dataframe and Data type dataframe
        '''
        logging.info('Reading data...')
        cnxn = pyodbc.connect(self.databasecon)   
        df = pd.read_sql(sql, cnxn)
        logging.info('Data read complete...')
        logging.info('Read: ' + str(df.shape[0]) + ' rows.')
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Count", "Column Type"]
        print(dtype_df)
        
        return df, dtype_df
    
    def getDtypes(self, data):
        dtype_df = data.dtypes.reset_index()
        dtype_df.columns = ["Count", "Column Type"]
        print(dtype_df)
        return dtype_df
    
    def splitData(self, df, trainsize):
        train, test = cv.train_test_split(df, train_size = trainsize)
        return train, test
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
   
    
    def closeLogger(self):
        logging.shutdown()
