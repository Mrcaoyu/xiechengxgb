# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:17:16 2017

@author: caoyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cross_validation import train_test_split

def build_feature(*data):
    product_info,cii_month=data
    train_data=pd.DataFrame()
    train_data['eval1']=product_info['eval']
    train_data['eval2']=product_info.eval2
    train_data['eval3']=product_info.eval3
    train_data['eval4']=product_info.eval4
    train_data['voters']=product_info.voters
    train_data['maxstock']=product_info.maxstock
    train_data['lat']=product_info.lat
    train_data['lon']=product_info.lon
    train_data.index=product_info.index
    cols=['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14','q15']
    for i,m in enumerate(cols):
        train_data[m]=cii_month.iloc[:,-1-i]
    return train_data
    

product_quantity=pd.read_csv("product_quantity.txt")
product_quantity.columns=['product_id', 'product_date', 'orderattribute1', 'orderattribute2',
       'orderattribute3', 'orderattribute4', 'ciiquantity', 'ordquantity',
       'price']
product_info=pd.read_csv("product_info.txt",index_col='﻿product_id')
#product_info.columns=['﻿product_id', 'district_id1', 'district_id2', 'district_id3',
#       'district_id4', 'lat', 'lon', 'railway', 'airport', 'citycenter',
#       'railway2', 'airport2', 'citycenter2', 'eval', 'eval2', 'eval3',
#       'eval4', 'voters', 'startdate', 'upgradedate', 'cooperatedate',
#       'maxstock']

product_quantity.sort_values(by=['product_id','product_date'],inplace=True)
product_quantity['product_month']=product_quantity['product_date'].apply(lambda x:x[:7])
cii_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()
ord_month=product_quantity.groupby(['product_id','product_month']).sum()['ordquantity'].unstack()
aver=cii_month.mean(axis=1)
#缺失值填补（均值填补）
for i in range(cii_month.shape[0]):
    cii_month.iloc[i,:]=cii_month.iloc[i,:].fillna(float(aver.iloc[i]))

#生成训练集
label=cii_month.iloc[:,-1]
data=build_feature(product_info,cii_month.iloc[:,:-1])
data=data[data.isnull().sum(axis=1)==0]

#分割训练集
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2,random_state=0)

xgtrain=xgb.DMatrix(x_train,label=y_train)
xgval=xgb.DMatrix(x_test,label=y_test)


params={}
params['objective']='reg:linear'
params['eta']=0.1
params['min_child_weight']=5
params['subsample']=0.7
params['colsample_bytree']=0.7
params['scale_pos_weight']=1
params['silent']=1
params['max_depth']=7
params['lambda']=0
params['eval_metric']='rmse'

watchlist=[(xgval,'val'),(xgtrain,'train')]
xgboost_model=xgb.train(params,xgtrain,num_boost_round=10000,evals=watchlist)

months=['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01',
        '2016-07-01','2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
#months=['2015-12-01','2016-01-01']
submission=pd.read_csv('prediction_lilei_20170320.txt')
col=['product_id','product_month','ciiquantity_month']
submission.columns=col
results=pd.DataFrame();
res=pd.DataFrame();
for month in months:
    feature_data=build_feature(product_info,cii_month)
    feature_data=feature_data[feature_data.isnull().sum(axis=1)==0]
    xgtest=xgb.DMatrix(feature_data)
    preds=xgboost_model.predict(xgtest,ntree_limit=xgboost_model.best_iteration)
    preds=pd.DataFrame(preds)
    preds=preds.set_index(feature_data.index)
    cii_month[month[:7]]=preds
    preds.columns=['ciiquantity_month']
    preds['product_id']=feature_data.index
    preds['product_month']=month
    results['product_id']=preds.product_id
    results['product_month']=preds.product_month
    results['ciiquantity_month']=preds.ciiquantity_month
    res=res.append(results)
results=pd.merge(submission,res,on=['product_id','product_month'],how='left').fillna(132)
results.drop(['ciiquantity_month_x'],axis=1,inplace=True)
results.columns=col
results.to_csv('xgbv1.txt',index=False)



