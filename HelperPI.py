# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:04:09 2017

@author: pairwin
"""

import pyodbc
import logging
import pandas as pd



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
        
    def readData(self, sql, temp_dir):
        '''
            Returns Dataframe and Data type dataframe
        '''
        logging.info('Reading data...')
        cnxn = pyodbc.connect(self.databasecon)   
        df = pd.read_sql(sql, cnxn)
        logging.info('Data read complete...')
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Count", "Column Type"]
        print(dtype_df)
        return df, dtype_df
   
    
    def closeLogger(self):
        logging.shutdown()