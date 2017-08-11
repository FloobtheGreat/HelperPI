# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:49:35 2017

@author: pairwin
"""

import sys
sys.path.insert(0, r"C:\users\pairwin\Documents\Github\HelperPI")

import HelperPI

helper = HelperPI.Helper()

sql = helper.getSQL(r'C:\users\pairwin\Desktop\testsql.sql')

data, dtypes = helper.readData(sql)

training, validation = helper.splitData(data, 0.8)
