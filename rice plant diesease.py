#!/usr/bin/env python
# coding: utf-8

# In[2]:


# HOLA!!!! 

#The dataset is in : https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases

# There is code tooo
# Paper : https://link.springer.com/chapter/10.1007/978-3-031-01984-5_7
# (But we don't have to stick to this ...... we can pick different techniques too)


# In[36]:


import cv2
import os
import numpy as np
import glob
from skimage.feature import greycomatrix, greycoprops
import pandas as pd


# In[37]:


features = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
ang = ['0','1','2','3']


# In[38]:


column_names = ['image']


# In[39]:


for angs in ang:
    for ft in features:
        column_names.append(ft +"_"+angs)


# In[40]:


column_names.append('label')


# In[41]:


df = pd.DataFrame(columns = column_names)


# In[42]:


main_dir = "rice_leaf_images/Healthy/"

for hi in os.listdir(main_dir):
    row =[]
    img_color = cv2.imread(main_dir+hi)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    median_applied = cv2.medianBlur(img, 3)
    ret, thresh1 = cv2.threshold(median_applied, 0, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)   
    glcm = greycomatrix(thresh1, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True, levels=256)
    row.append(hi)
#     print(hi)
    for ft in features:
#         print(ft)
#         print(greycoprops(glcm, ft)[0])
        for i in range(4):
#             print(i)
            value = greycoprops(glcm, ft)[0][i]
#             print(value)
            row.append(value)
    row.append(1)
    df.loc[len(df.index)] = row


# In[43]:


unhealthy_dir = "rice_leaf_images/Unhealthy/all/"

for ui in os.listdir(unhealthy_dir):
    row =[]
    img_color = cv2.imread(unhealthy_dir+ui)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    median_applied = cv2.medianBlur(img, 3)
    ret, thresh1 = cv2.threshold(median_applied, 0, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)   
    glcm = greycomatrix(thresh1, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True, levels=256)
    row.append(ui)
#     print(ui)
    for ft in features:
#         print(ft)
#         print(greycoprops(glcm, ft)[0])
        for i in range(4):
#             print(i)
            value = greycoprops(glcm, ft)[0][i]
#             print(value)
            row.append(value)
    row.append(0)
    df.loc[len(df.index)] = row


# In[46]:


df.to_csv('Feature_1.csv',index=False)


# In[ ]:





# In[ ]:




