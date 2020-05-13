#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import keras
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('Desktop/DiabetesX.csv')
df_y = pd.read_csv('Desktop/DiabetesY.csv')
y_train = df_y['Outcome'].values


# In[3]:


pr = df['Pregnancies'].values
gl = df['Glucose'].values
bp = df['BloodPressure'].values
st = df['SkinThickness'].values
ins = df['Insulin'].values
bmi = df['BMI'].values
dpf = df['DiabetesPedigreeFunction'].values
age = df['Age'].values


# In[4]:


def get_data(pr, gl, bp, st, ins, bmi, dpf, age):
    x_train = []
    for i in range(0, len(pr)):
        x = []
        x.append(pr[i])
        x.append(gl[i])
        x.append(bp[i])
        x.append(st[i])
        x.append(ins[i])
        x.append(bmi[i])
        x.append(dpf[i])
        x.append(age[i])
        x_train.append(x)
    return x_train


# In[5]:


df_t = pd.read_csv('Desktop/Diabetes_Xtest.csv')


# In[6]:


prt = df_t['Pregnancies'].values
glt = df_t['Glucose'].values
bpt = df_t['BloodPressure'].values
stt = df_t['SkinThickness'].values
inst = df_t['Insulin'].values
bmit = df_t['BMI'].values
dpft = df_t['DiabetesPedigreeFunction'].values
aget = df_t['Age'].values


# In[7]:


x_test = get_data(prt,glt, bpt,stt,inst,bmit,dpft,aget)
x_train = get_data(pr,gl,bp,st,ins,bmi,dpf,age)


# In[8]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn = knn.fit(x_train, y_train)


# In[9]:


knn.predict(x_test)

