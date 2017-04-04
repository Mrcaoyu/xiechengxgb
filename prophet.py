# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:35:27 2017

@author: caoyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

product_quantity=pd.read_csv("product_quantity.txt")
product_quantity.columns=['product_id', 'product_date', 'orderattribute1', 'orderattribute2',
       'orderattribute3', 'orderattribute4', 'ciiquantity', 'ordquantity',
       'price']
product_info=pd.read_csv("product_info.txt",index_col='﻿product_id')

product_quantity.sort_values(by=['product_id','product_date'],inplace=True)
product_quantity['product_month']=product_quantity['product_date'].apply(lambda x:x[:7])
cii_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()
ord_month=product_quantity.groupby(['product_id','product_month']).sum()['ordquantity'].unstack()
aver=cii_month.mean(axis=1)
#缺失值填补（均值填补）
for i in range(cii_month.shape[0]):
    cii_month.iloc[i,:]=cii_month.iloc[i,:].fillna(float(aver.iloc[i]))
m=Prophet()
res=pd.DataFrame()
months=['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01',
        '2016-07-01','2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
for i in range(cii_month.shape[0]):
    df=cii_month.iloc[i,:].reset_index()
    df.columns=['ds','y']
    m.fit(df)
    future=m.make_future_dataframe(periods=14,freq='M')
    forecast=m.predict(future)
    re=forecast.iloc[-14:,-1]
    re=pd.DataFrame(re)
    re=re.reset_index()
    re.drop(['index'],axis=1,inplace=True)
    re.insert(0,'product_id',cii_month.index[i])
    re.insert(1,'product_month',months)
    re.rename(columns={'yhat':'ciiquantity_month'},inplace=True)
    res=res.append(re)
    