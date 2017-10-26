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
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
import scipy.stats as stats
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import shutil
import csv
import math
from operator import itemgetter
from time import time
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    
    def readDataToCSV(self, sql, directory):
        logging.info('Writing data to temp csv...')
        filename = directory+'\\file.csv'
        cnxn = pyodbc.connect(self.databasecon)
        cursor = cnxn.cursor()
        cursor.execute(sql)
        
        with open(filename, 'a+') as f:
            writ = csv.writer(f)
            for row in  cursor.fetchall():
                writ.writerow(str(row))
                
        logging.info('Finished writing data to ' + filename)
        return filename
    
    def makeTempDir(self):
        logging.info('Making Temp Directory...')
        tmp = tempfile.mkdtemp()
        return tmp
    
    def deleteTemp(self, temp_dir):
        logging.info('Deleting Temp Directory...')
        try:
            shutil.rmtree(temp_dir) 
        except:
            logging.info('Directory already deleted!.')
    
    def getDtypes(self, data):
        logging.info('Getting data types...')
        dtype_df = data.dtypes.reset_index()
        dtype_df.columns = ["Count", "Column Type"]
        print(dtype_df)
        return dtype_df
    
    def splitData(self, df, trainsize):
        """
            Splits a single DataFrame into a train and test frame.
        """
        train, test = cv.train_test_split(df, train_size = trainsize)
        return train, test
    
    def splitData2(self, X, y, testsize=0.3, random_state=0):
        """
            Takes an X and y dataframe and returns 4 np arrays
        """
        train_X, test_X, train_y, test_y = cv.train_test_split(X, y, test_size=testsize, random_state=random_state)
        return  train_X, test_X, train_y, test_y
    
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
        

    def report(self, gs):
        """
        Utility function to report best scores
        from a grid search
        """
        print("Best Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  gs.best_score_,
                  np.std(gs.cv_results_['mean_test_score'])))
        print("Parameters: {0}".format(gs.best_params_))
       
        
    def calc_VIFs(self, X):
        vif = pd.DataFrame()
        vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['feature'] = X.columns
        
        vif.sort_values(by='VIF Factor', ascending=False, inplace=True)
        return vif
    
    
    
    def find_iteractions(self, X, y):
         result = list()
         for var1 in X.columns:
             for var2 in X.columns:
                 if var1 != var2:
                     vardf = X[var1]*X[var2]
                     coef, pval = stats.pearsonr(vardf.values, y)
                     if pval <= 0.05:
                         lin = LinearRegression()
                         lin.fit(vardf.values.reshape(-1,1), y)
                         ytrue = y
                         ypred = lin.predict(vardf.values.reshape(-1,1))
                         r2 = r2_score(ytrue, ypred)
                         result.append([var1+'*'+var2,coef, pval, r2])
                     
         final = pd.DataFrame(result, columns = ['name','coef','pval','r2'])
         print(final)

        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
 
=======
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(dtrain.values, label=dlabels.values, feature_names = feature_names)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            alg.set_params(n_estimators=cvresult.shape[0])
    
        alg.fit(dtrain, dlabels, eval_metric='auc')
    
        dtrain_predictions = alg.predict(dtrain)
        dtrain_predprob = alg.predict_proba(dtrain)[:,1]
    
        #Print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % accuracy_score(dlabels.values, dtrain_predictions))
        print("AUC Score (Train): %f" % roc_auc_score(dlabels.values, dtrain_predprob))
    
        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        
    def find_iteractions(self, X, y):
        result = list()
        for var1 in X.columns:
            for var2 in X.columns:
                if var1 != var2:
                    vardf = X[var1]*X[var2]
                    coef, pval = stats.pearsonr(vardf.values, y)
                    if pval <= 0.05:
                        lin = LinearRegression()
                        lin.fit(vardf.values.reshape(-1,1), y)
                        ytrue = y
                        ypred = lin.predict(vardf.values.reshape(-1,1))
                        r2 = r2_score(ytrue, ypred)
                        result.append([var1+'*'+var2,coef, pval, r2])
                    
        final = pd.DataFrame(result, columns = ['name','coef','pval','r2'])
        print(final)
>>>>>>> de5936f2cb3cfd5b37b0250e74ee81db1e0bd7c9
