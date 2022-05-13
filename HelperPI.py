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
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import scipy.stats as stats
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
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
        self.databasecon = r'DRIVER={NetezzaSQL};SERVER=SRVDWHITP04;DATABASE=EDW_SPOKE;UID=pairwin;PWD=pairwin;TIMEOUT=0'        
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
    
    
    
    def find_interactions(self, X, y, alpha=95, gamma=0.2):
         result = list()
         for var1 in X.columns:
             for var2 in X.columns:
                 if var1 != var2:
                     vardf = X[var1]*X[var2]
                     coef, pval = stats.pearsonr(vardf.values, y)
                     if pval <= (1-alpha)/100 and coef >= gamma:
                         lin = LinearRegression()
                         lin.fit(vardf.values.reshape(-1,1), y)
                         ytrue = y
                         ypred = lin.predict(vardf.values.reshape(-1,1))
                         r2 = r2_score(ytrue, ypred)
                         result.append([var1+'*'+var2,coef, pval, r2])
                     
         return pd.DataFrame(result, columns = ['name','coef','pval','r2'])
         
         
         
    def elbow_method(self, df):
        kmeans_kwargs = {
         "init": "random",
         "n_init": 10,
         "max_iter": 300,
         "random_state": 42}
        
        # A list holds the SSE values for each k
        sse = []
        
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(df)
            sse.append(kmeans.inertia_)
        
        #plt.style.use("fivethirtyeight")
        plt.plot(range(2, 11), sse)
        plt.xticks(range(2, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()
        
    def silhouette_method(self, df):
        sil_output = []
        
        for i in range(2,11):
            kmeans = KMeans(n_clusters=i, random_state=42).fit(df)
            sil_output.append(silhouette_score(df, kmeans.labels_))
        
        plt.plot(range(2, 11), sil_output)
        plt.xticks(range(2, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.show()      
        
    def optimized_k(self, df, k_min=2, k_max=11):
        from sklearn.preprocessing import minmax_scale
        from sklearn.cluster import KMeans
        kmeans_kwargs = {
         "init": "random",
         "n_init": 10,
         "max_iter": 300,
         "random_state": 42}
        
        # A list holds the SSE values for each k
        MD = [] #Mean Distortion
        df_n = df.shape[0]
        alpha_min = 3.14
        k_opt = 0
        
        for k in range(k_min, k_max):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(df)
            MD.append((kmeans.inertia_/df_n))
        
        MD_Scaled = minmax_scale(MD, (0,10))
        
        plt.plot(range(k_min, k_max), MD_Scaled)
        plt.xticks(range(k_min, k_max))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Normalized Mean Distortion")
        plt.show()
        
        for i in range(0, k_max-k_min-2):
            j=i+1
            k=i+2
            
            P_i = MD_Scaled[i]
            P_j = MD_Scaled[j]
            P_k = MD_Scaled[k]
            
            a = np.linalg.norm(P_i-P_j)
            b = np.linalg.norm(P_j-P_k)
            c = np.linalg.norm(P_k-P_i)
            
            alpha = np.arccos((a**2 * b**2 * c**2)/(2*a*b))
            
            if alpha < alpha_min:
                alpha_min = alpha
                k_opt = j
        
        print('Optimum K =', str(k_opt))
        return (alpha_min, k_opt)
    
         
    def res_v_fit_plot(self, results):
        '''
        

        Parameters
        ----------
        results : Regression Result
            takes a regression result and plots Residuals vs Fitted values.

        Returns
        -------
        None.

        '''
        residuals = results.resid
        fitted = results.fittedvalues
        smoothed = lowess(residuals,fitted)
        top3 = abs(residuals).sort_values(ascending = False)[:3]
        
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["figure.figsize"] = (8,7)
        fig, ax = plt.subplots()
        ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
        ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax.set_ylabel('Residuals')
        ax.set_xlabel('Fitted Values')
        ax.set_title('Residuals vs. Fitted')
        ax.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)
        
        for i in top3.index:
            ax.annotate(i,xy=(fitted[i],residuals[i]))
        
        plt.show()
    
    def qq_plot(self, results):
        '''
        

        Parameters
        ----------
        results : OLS Regression Result
            Takes only OLS regression result and plots QQ Normality plot.

        Returns
        -------
        None.

        '''
        sorted_student_residuals = pd.Series(results.get_influence().resid_studentized_internal)
        sorted_student_residuals.index = results.resid.index
        sorted_student_residuals = sorted_student_residuals.sort_values(ascending = True)
        df = pd.DataFrame(sorted_student_residuals)
        df.columns = ['sorted_student_residuals']
        df['theoretical_quantiles'] = stats.probplot(df['sorted_student_residuals'], dist = 'norm', fit = False)[0]
        rankings = abs(df['sorted_student_residuals']).sort_values(ascending = False)
        top3 = rankings[:3]
        
        fig, ax = plt.subplots()
        x = df['theoretical_quantiles']
        y = df['sorted_student_residuals']
        ax.scatter(x,y, edgecolor = 'k',facecolor = 'none')
        ax.set_title('Normal Q-Q')
        ax.set_ylabel('Standardized Residuals')
        ax.set_xlabel('Theoretical Quantiles')
        ax.plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')
        for val in top3.index:
            ax.annotate(val,xy=(df['theoretical_quantiles'].loc[val],df['sorted_student_residuals'].loc[val]))
        plt.show()
        
    def scale_location(self, results):
        '''
        

        Parameters
        ----------
        results : OLS Regression Result
            Takes only OLS regression result and plots Scale-Location Chart .

        Returns
        -------
        None.

        '''
        fitted = results.fittedvalues
        student_residuals = results.get_influence().resid_studentized_internal
        sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
        sqrt_student_residuals.index = results.resid.index
        smoothed = lowess(sqrt_student_residuals,fitted)
        top3 = abs(sqrt_student_residuals).sort_values(ascending = False)[:3]
        
        fig, ax = plt.subplots()
        ax.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
        ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
        ax.set_xlabel('Fitted Values')
        ax.set_title('Scale-Location')
        ax.set_ylim(0,max(sqrt_student_residuals)+0.1)
        for i in top3.index:
            ax.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
        plt.show()
        
    def res_vs_leverage(self, results):
        '''
        

        Parameters
        ----------
        results : OLS Regression Result
            Takes only OLS regression result and plots Residuals vs Leverage.

        Returns
        -------
        None.

        '''
        student_residuals = pd.Series(results.get_influence().resid_studentized_internal)
        student_residuals.index = results.resid.index
        df = pd.DataFrame(student_residuals)
        df.columns = ['student_residuals']
        df['leverage'] = results.get_influence().hat_matrix_diag
        smoothed = lowess(df['student_residuals'],df['leverage'])
        sorted_student_residuals = abs(df['student_residuals']).sort_values(ascending = False)
        top3 = sorted_student_residuals[:3]
        
        fig, ax = plt.subplots()
        x = df['leverage']
        y = df['student_residuals']
        xpos = max(x)+max(x)*0.01  
        ax.scatter(x, y, edgecolors = 'k', facecolors = 'none')
        ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax.set_ylabel('Studentized Residuals')
        ax.set_xlabel('Leverage')
        ax.set_title('Residuals vs. Leverage')
        ax.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
        ax.set_xlim(-0.01,max(x)+max(x)*0.05)
        plt.tight_layout()
        for val in top3.index:
            ax.annotate(val,xy=(x.loc[val],y.loc[val]))
        
        cooksx = np.linspace(min(x), xpos, 50)
        p = len(results.params)
        poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
        poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
        negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
        negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)
        
        ax.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
        ax.plot(cooksx,poscooks05y, ls = ':', color = 'r')
        ax.plot(cooksx,negcooks1y, ls = ':', color = 'r')
        ax.plot(cooksx,negcooks05y, ls = ':', color = 'r')
        ax.plot([0,0],ax.get_ylim(), ls=":", alpha = .3, color = 'k')
        ax.plot(ax.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
        ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
        ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
        ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
        ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
        ax.legend()
        plt.show()