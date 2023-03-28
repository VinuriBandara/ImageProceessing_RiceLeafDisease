import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops

import pandas as pd
import pickle
import matplotlib.pyplot as plt


loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

features = ['contrast','dissimilarity','homogeneity','energy','correlation']
ang = ['0','1','2','3']

column_names =['image']

for angs in ang:
    for ft in features:
        column_names.append(ft +"_"+angs)

def test_image_dist1(image):
    row =[]
    img_color = cv2.imread(image)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    median_applied = cv2.medianBlur(img, 3)
    print("Now applying median filtering to "+image)
    ret, thresh1 = cv2.threshold(median_applied, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    print("Now applying image segmentation to "+image)
    
    glcm = graycomatrix(thresh1, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True, levels=256)
    print("Now applying feature extraction to "+image)
    row.append(image)
    for ft in features:

        for i in range(4):
            value = graycoprops(glcm, ft)[0][i]

            row.append(value)
    df1.loc[len(df1.index)] = row
    return df1

def test_image_dist2(image):
    row =[]
    img_color = cv2.imread(image)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    median_applied = cv2.medianBlur(img, 3)
    ret, thresh1 = cv2.threshold(median_applied, 0, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)   
    glcm = graycomatrix(thresh1, distances=[2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True, levels=256)

    row.append(image)
    for ft in features:
        for i in range(4):
            value = graycoprops(glcm, ft)[0][i]
            row.append(value)
    df2.loc[len(df2.index)] = row
    return df2


result_images = []

for images in os.listdir("Testing/"):

    image = "Testing/"+images

    print("Classifying "+image)

    df1 = pd.DataFrame(columns = column_names)
    df2 = pd.DataFrame(columns = column_names)
    
    df1= test_image_dist1(image)

    df2 = test_image_dist2(image)
    
    df1 = df1.add_suffix("_d1")
    df2 = df2.add_suffix("_d2")
    
    df = df1.merge(df2,left_on="image_d1",right_on="image_d2")

    df = df.drop(columns=['image_d2'])

    df = df.drop(columns=['image_d1'],axis=1)
    
    print("Now predicting disease status of "+image)

    result = loaded_model.predict(df)

    plt.figure(figsize = (12,12))
    

    if (result == 1):
        print("This plant is healthy \n")
        img = cv2.imread(image)
        plt.imshow(img)
        plt.text(10, 12, "THIS IS A HEALTHY PLANT", fontsize=14, color='blue')

        plt.show(block=False)
        plt.pause(5)


    if (result == 0):
        print("This plant is unhealthy \n")
        img = cv2.imread(image)
        plt.imshow(img)
        plt.text(10, 12, "THIS IS AN UNHEALTHY PLANT", fontsize=14, color='red')

        plt.show(block=False)
        plt.pause(5)
 
