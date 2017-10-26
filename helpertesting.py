# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:49:35 2017

@author: pairwin
"""

import sys
sys.path.insert(0, r"C:\users\pairwin\Documents\Github\HelperPI")
import HelperPI
import pandas as pd
helper = HelperPI.Helper()


base = pd.read_csv(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\SQL\tab_model.csv', parse_dates=['DATE_VALUE'])

collist_cont = ['MEDHINC_CY','MEDAGE_CY','CLOSEST_BP','CLOSEST_CAB','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD',
                'DAYS_AS_CUSTOMER','TOTAL_TRANSACTIONS','REW_TRANSACTIONS','TOTAL_SPEND','FISH_SALES','HUNT_SALES',
                'OTHER_SALES','TRANSACTIONS_QMIN1','SPEND_QMIN1','TRANSACTIONS_QMIN2','SPEND_QMIN2','TRANSACTIONS_QMIN3',
                'SPEND_QMIN3','TRANSACTIONS_QMIN4','SPEND_QMIN4','FISH_SALES_QMIN1','HUNT_SALES_QMIN1','OTHER_SALES_QMIN1',
                'FISH_SALES_QMIN2','HUNT_SALES_QMIN2','OTHER_SALES_QMIN2','FISH_SALES_QMIN3','HUNT_SALES_QMIN3',
                'OTHER_SALES_QMIN3','FISH_SALES_QMIN4','HUNT_SALES_QMIN4','OTHER_SALES_QMIN4','DAYS_SINCE_PURCHASE',
                'AVERAGE_TICKET','AVERAGE_TICKET_QMIN1', 'AVERAGE_TICKET_QMIN2','AVERAGE_TICKET_QMIN3','AVERAGE_TICKET_QMIN4',
                'DAYS_BTWN_PURCHASE','IND_IN_HH']


testing = base[collist_cont]
y = base['TARGET_PURCH_NEXT15']

for var in collist_cont:
    print(var)
    vardf = testing[var]
    col, coef, _ = helper.pbiserial_transform(vardf, y, var)
    print (col + ' ' + str(coef))
    print('\n')
