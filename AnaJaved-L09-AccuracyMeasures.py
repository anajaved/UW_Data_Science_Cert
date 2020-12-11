#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anajaved

Since I am builing on previous code submitted, lesson 9 Code Changes
begin on line 400.
"""


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
1) Short narrative on the data preparation for your chosen data set 
     from Milestone 2.
     
The number of observations in this dataset are 155.
The number of attributes in this dataset are 19.

The following is a breakdown of the data types, distribution of the attributes 
used in this assignment:
    The "Alive_Dead" column is a categorical attribute with 123 patients alive, and 32 dead. 
    The "Age" column is numeric, and has no outliers and is a bimodal distribution 
    The "Bilirubin" column is numeric and appears to have an exponential decay distribution
    The "ALK_Phosphate" column is numeric and appears to have an exponential decay distribution
    The "SGOT" column is numeric and appears to have an exponential decay distribution    
    The "Albumin" column is numeric and has a normal distribution.
    The "Antivirals" column is categorical and 24 are not on antivirals, and 131 are.
    The "Sex" column is categorical and 139 are male, and 16 are female. 
    The "Protime" column is numeric and appears to be normally distributed. 

The data was obtained from the following website. It is a public dataset provided
by the University of California, Irving. 
https://archive.ics.uci.edu/ml/datasets/hepatitis

From this data, I would like to ask the following questions:
        - Does your Albumin level have an effect on chances of living when you are 
            diagnosed with hepatitis? 
            
        - At what threshold do SGOT levels get dangerous (or risk life)
            for patients diagnosed with hepatitis? 
"""

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------  
"""
2) Import statements for necessary package(s).
"""

## Importing Required Packages for this assignment: 
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from copy import deepcopy


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
   3) Read in the dataset from a freely and easily available source on the 
    internet.
        
"""
## Reading in the Hepatitis dataset from University of California Irvine 
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
Hep = pd.read_csv(url, header = None)

## Assigning Reasonable Column Names 
Hep.columns = ['Alive_Dead', 'Age', 'Sex', 'Steroids', 'Antivirals',
               'Fatigue', 'Malaise', 'Anorexia', 'Large_liver', 'Firm_liver',
               'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin',
               'ALK_Phosphate', 'SGOT', 'Albumin', 'Protime', 'Histology'
               ]
    
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

"""
4) Show data preparation. Normalize some numeric columns, one-hot encode 
    some categorical columns with 3 or more categories, remove or replace 
    missing values, remove or replace some outliers.
"""
    
# Removing & Replacing missing values with attribute Median 
#  Columns with missing values: 
# Bilirubin, ALK_Phosphate, SGOT, Albumin, Protime

Hep.loc[:, "Bilirubin"] = pd.to_numeric(Hep.loc[:, "Bilirubin"], errors='coerce')
HasNan1 = np.isnan(Hep.loc[:, "Bilirubin"] )
sum(HasNan1)  # 6
Hep.loc[HasNan1, "Bilirubin"] = np.nanmedian(Hep.loc[:, "Bilirubin"] )

Hep.loc[:, "ALK_Phosphate"] = pd.to_numeric(Hep.loc[:, "ALK_Phosphate"], errors='coerce')
HasNan2 = np.isnan(Hep.loc[:, "ALK_Phosphate"])
sum(HasNan2) # 29
Hep.loc[HasNan2, "ALK_Phosphate"] = np.nanmedian(Hep.loc[:, "ALK_Phosphate"] )

Hep.loc[:, "SGOT"] = pd.to_numeric(Hep.loc[:, "SGOT"], errors='coerce')
HasNan3 = np.isnan(Hep.loc[:, "SGOT"])
sum(HasNan3) # 4
Hep.loc[HasNan3, "SGOT"] = np.nanmedian(Hep.loc[:, "SGOT"] )

Hep.loc[:, "Albumin"] = pd.to_numeric(Hep.loc[:, "Albumin"], errors='coerce')
HasNan4 = np.isnan(Hep.loc[:, "Albumin"])
sum(HasNan4) # 16 
Hep.loc[HasNan4, "Albumin"] = np.nanmedian(Hep.loc[:, "Albumin"] )

Hep.loc[:, "Protime"] = pd.to_numeric(Hep.loc[:, "Protime"], errors='coerce')
HasNan5 = np.isnan(Hep.loc[:, "Protime"])
sum(HasNan5) # 67
Hep.loc[HasNan5, "Protime"] = np.nanmedian(Hep.loc[:, "Protime"] )


## Removing & Replacing Outliers 

# plt.hist(Hep.loc[:, "Alive_Dead"])   # No Outliers, just 1 or 2 
# plt.hist(Hep.loc[:, "Age"])          # No Outliers, bimodal distribution
# plt.hist(Hep.loc[:, "Bilirubin"])    # Has Outliers. (5+)
# plt.hist(Hep.loc[:, "ALK_Phosphate"])# Has Outliers 200+
# plt.hist(Hep.loc[:, "SGOT"])         # Has Outliers 300+
# plt.hist(Hep.loc[:, "Albumin"])      # Has 5.5+ 
# plt.hist(Hep.loc[:, "Sex"])          # No Outliers, just 1 or 2 
# plt.hist(Hep.loc[:, "Antivirals"])   # No Outliers, just 1 or 2 

TooHigh1 = Hep.loc[:, "Bilirubin"] > 4 # Outlier
Hep.loc[TooHigh1, "Bilirubin"] = 1 # Most Common Value

TooHigh2 = Hep.loc[:, "ALK_Phosphate"] > 200 # Outlier
Hep.loc[TooHigh2, "ALK_Phosphate"] = 100 # Most Common Value

TooHigh3 = Hep.loc[:, "SGOT"] > 300 # Outlier
Hep.loc[TooHigh3, "SGOT"] = 50 # Most Common Value

TooHigh4 = Hep.loc[:, "Albumin"] > 5 # Outlier
Hep.loc[TooHigh4, "Albumin"] = 4 # Most Common Value


## Binning Age column into a categorical variable 
Hep.loc[:, "Age_Decade"] = Hep.loc[:, "Age"]

# Creating the new categories based on age: 10s, 20s, 30s, 40s, 50s...etc. 
for each in Hep.loc[:, "Age_Decade"].unique():
    # print(each)
    if len(str(each)) >1:
        age_bucket = str(each)[0] + '0s'
        Hep.loc[ Hep.loc[:, "Age_Decade"] == each, "Age_Decade"] = age_bucket
    elif len(str(each)) <= 1:
        age_bucket = '10s'
        Hep.loc[ Hep.loc[:, "Age_Decade"] == each, "Age_Decade"] = age_bucket
        
# Consolidating the following categories in "Age_Decade"
Hep.loc[Hep.loc[:, "Age_Decade"] == "60s", "Age_Decade"] = "60+"
Hep.loc[Hep.loc[:, "Age_Decade"] == "70s", "Age_Decade"] = "60+"
Hep.loc[Hep.loc[:, "Age_Decade"] == "10s", "Age_Decade"] = "<20"

### For Missing values in Category columns, assigning them to most popular value 

# Hep.loc[:, "Steroids"].value_counts()
Replace = Hep.loc[:, "Steroids"] == "?"    # 1 unknown
Hep.loc[Replace, "Steroids"] = "1"

# Hep.loc[:, "Fatigue"].value_counts()        # 1 unknown
Replace = Hep.loc[:, "Fatigue"] == "?"
Hep.loc[Replace, "Fatigue"] = "1"

# Hep.loc[:, "Malaise"].value_counts()         # 1 unknown
Replace = Hep.loc[:, "Malaise"] == "?"
Hep.loc[Replace, "Malaise"] = "2"

# Hep.loc[:, "Anorexia"].value_counts()           # 1 unknown
Replace = Hep.loc[:, "Anorexia"] == "?"
Hep.loc[Replace, "Anorexia"] = "2"

# Hep.loc[:, "Large_liver"].value_counts()        # 10 unknowns
Replace = Hep.loc[:, "Large_liver"] == "?"
Hep.loc[Replace, "Large_liver"] = "2"

# Hep.loc[:, "Firm_liver"].value_counts()         # 11 unknowns
Replace = Hep.loc[:, "Firm_liver"] == "?"
Hep.loc[Replace, "Firm_liver"] = "2"

# Hep.loc[:, "Spleen_Palpable"].value_counts()    # 5 unknown 
Replace = Hep.loc[:, "Spleen_Palpable"] == "?"
Hep.loc[Replace, "Spleen_Palpable"] = "2"

# Hep.loc[:, "Spiders"].value_counts()         # 5 unknown 
Replace = Hep.loc[:, "Spiders"] == "?"
Hep.loc[Replace, "Spiders"] = "2"

# Hep.loc[:, "Ascites"].value_counts()         # 5 unknown 
Replace = Hep.loc[:, "Ascites"] == "?"
Hep.loc[Replace, "Ascites"] = "2"

# Hep.loc[:, "Varices"].value_counts()        # 5 unknown 
Replace = Hep.loc[:, "Varices"] == "?"
Hep.loc[Replace, "Varices"] = "2"

#Hep.loc[:, "Histology"].value_counts()      # none 


## Constructing a new categorical variables by hot encoding 

Hep.loc[:, "<20"] = (Hep.loc[:, "Age_Decade"] == "<20").astype(int)
Hep.loc[:, "20s"] = (Hep.loc[:, "Age_Decade"] == "20s").astype(int)
Hep.loc[:, "30s"] = (Hep.loc[:, "Age_Decade"] == "30s").astype(int)
Hep.loc[:, "40s"] = (Hep.loc[:, "Age_Decade"] == "40s").astype(int)
Hep.loc[:, "50s"] = (Hep.loc[:, "Age_Decade"] == "50s").astype(int)
Hep.loc[:, "60+"] = (Hep.loc[:, "Age_Decade"] == "60+").astype(int)

## Normalizing Numeric Categories
## Will use the "Albumin" Column. Will first convert it to float64 type 

Hep.loc[:, "Albumin"] = pd.to_numeric(Hep.loc[:, "Albumin"], errors='coerce')
# HasNan4 = np.isnan(Hep.loc[:, "Albumin"])
# sum(HasNan4) # 16 Unknowns. I will leave them alone for now 

X= Hep.loc[:, "Albumin"]
X= pd.DataFrame(X) 

# Using sklearn standardization_scale 
standardization_scale = StandardScaler().fit(X)
z = standardization_scale.transform(X)

# ## Before Z-Normalization Plot
# plt.hist(Hep.loc[:, "Albumin"], bins = 20, color='orange')
# plt.title("Before Z-normalization - Albumin")
# plt.show()

# ## After Z-Normalization Plot 
# plt.hist(z, bins = 20, color='green')
# plt.title("After Z-normalization  - Albumin")
# plt.show()

## Assigning the normalized numeric values to the original Dataframe
Hep.loc[:, "Albumin"] = z

###### Will also normalize SGOT Column

# Hep.dtypes
Y = Hep.loc[:, "SGOT"]
Y = pd.DataFrame(Y) 

# Using sklearn standardization_scale 
standardization_scale = StandardScaler().fit(Y)
z = standardization_scale.transform(Y)

## Assigning the normalized numeric values to the original Dataframe
Hep.loc[:, "SGOT"] = z

##### Will also normalize Age Column

Hep.dtypes
Z = Hep.loc[:, "Age"]
Z = pd.DataFrame(Z) 

# Using sklearn standardization_scale 
standardization_scale = StandardScaler().fit(Z)
z = standardization_scale.transform(Z)

## Assigning the normalized numeric values to the original Dataframe
Hep.loc[:, "Age"] = z

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
5) Ask a binary-choice question that describes your classification. 
     Write the question as a comment. Specify an appropriate column as 
     your expert label for a classification (include decision comments).
     
The binary-choice (yes/no) question I would like to ask is 
    Does your Albumin level have an effect on chances of living when you are diagnosed 
    with hepatitis? 
    
    The column that will serve as the expert lable for classification is 
    "Alive_Dead". The following is the breakdown in values: 
    1 = Dead
    2 = Alive 
"""
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

"""
   6) Apply K-Means on some of your columns, but make sure you do not use the 
        expert label. Add the K-Means cluster labels to your dataset.
"""
## Clustering Data with sklearn, prior to normalization 
X = Hep.loc[:,["Albumin"]] 
kmeans = KMeans(init='random', n_clusters=2, random_state=0).fit(X)  
Labels = kmeans.labels_
ClusterCentroids = kmeans.cluster_centers_
# print(ClusterCentroids)
# print(Labels)

# Adding Add the K-Means cluster label to the dataset.
X.loc[:,'Labels'] = Labels
Hep.loc[:,'Labels'] = Labels

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

"""
   7) Split your data set into training and testing sets using the proper 
        function in sklearn (include decision comments).
"""
x = Hep.loc[:,["Alive_Dead", "Albumin", "Labels"]]

x_train, x_test, y_train, y_test = train_test_split (np.array(Hep.loc[:,"Albumin"]),
                                                     np.array(Hep.loc[:,"Alive_Dead"]),
                                                     test_size = 0.5,
                                                     random_state = 123)

# I chose a test size of 0.5, or half since I wanted the testing and training 
# data to be the same. I did not want to risk over or under fitting. 
# I specified a random_state since I wanted the results to be reproducible. 

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
   8) Create a classification model for the expert label based on the training 
        data (include decision comments).
"""
# I will be using the Naive Bayes classifier
nbc = GaussianNB(priors=[0.4, 0.6])
#nbc = GaussianNB()

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
9) Apply your (trained) classifiers to the test data to predict 
     probabilities.
"""
nbc.fit(x_train.reshape(-1,1), y_train)
print ("Predictions for test set:")
print (nbc.predict(x_test.reshape(-1,1)))
print ('Actual class values:')
print (y_test)
print ('Probability of test vector')
print(nbc.predict_proba(x_test.reshape(-1,1))) #     Return probability estimates for the test vector X.

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
10) Write out to a csv a dataframe of the test data, including actual 
     outcomes, and the probabilities of your classification.
"""
# Creating a new data frame to store the test data, actual, predicted, 
# and probabilities 
csv_data = pd.DataFrame()
csv_data.loc[:, "test_data"] = x_test
csv_data.loc[:, "actual"] = y_test
csv_data.loc[:, "predictions"] = nbc.predict(x_test.reshape(-1,1))

# There are two values, probability of living and probably of dying, going to separate out  
probabilities = nbc.predict_proba(x_test.reshape(-1,1)) 

csv_data.loc[:, "probabilities - Dead"] = np.take(probabilities, 0, axis = 1)
csv_data.loc[:, "probabilities - Alive"] = np.take(probabilities, 1, axis = 1)

# Creating CSV 
# csv_data.to_csv(#path_or_buf='/Users/anajaved/Desktop/AnaJaved-L08-SupervisedLearning.csv',
#            sep=',', index=False)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
11) Determine accuracy rate, which is the number of correct predictions 
     divided by the total number of predictions (include brief preliminary 
                                                 analysis commentary).
"""
# Boolean array of when actual results matched predictions from model: 
matched_results = csv_data.loc[:,'actual'] == csv_data.loc[:,'predictions'] 

# Calculating Accuracy Rate 
# print("Accuracy Rate (%)")
# print((sum(matched_results) / len(matched_results))*100)



# ---------------------------------------------------------------------
#################### LESSON 9 ASSIGNMENT:##############################
# ---------------------------------------------------------------------
"""
Build on your previous model-building Python script. 
The Python script must run in the grader's environment using only spyder's run file button.  Add the following:

"""

from sklearn.metrics import *
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
 1.) Confusion Matrix from your predicted values 
(specify and justify the probability threshold)."""

target = np.array(csv_data.loc[:, "actual"])  # Assigning actual data to variable
predictions =  np.array(csv_data.loc[:, "predictions"]) # Assigning predictions data to variable
probabilities = np.array(csv_data.loc[:, "probabilities - Alive"])

# Passing Target and Prediction data to Confusion_Matrix
CM = confusion_matrix(target, predictions)
print ("\n\nConfusion matrix:\n", CM)

tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)

# The probability threshold was specified above when calling the 
# Naive Bayes Classifiers (line 340) 
# For class 0, I applied a 0.4 threshold. For class 1, I applied a 0.6 threshold. 
# This is because this provided the highest accuracy for the predictions. 


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
"""
2.) Precision, Recall, and F1 measures based on your Confusion Matrix.
"""

P = precision_score(target, predictions)
print ("\nPrecision Score:", np.round(P, 2))
print (' ')
R = recall_score(target, predictions)
print ("\nRecall Score:", np.round(R, 2))
print (' ')
F1 = f1_score(target, predictions)
print ("\nF1 Score:", np.round(F1, 2))
print (' ')

"""
3.) Calculate the ROC curve and it's AUC using sklearn. 
     Present the ROC curve. Present the AUC in the ROC's plot.
"""

# ROC analysis
LC = 'black' # Line Color for plot 
LW = 1.5 # line width for plots
LL = "lower right" # legend location for plot
LC = 'red' # Line Color for ROC plot 


fpr, tpr, th = roc_curve(target, predictions, pos_label=2) # 2 = Alive, 1 = Dead 
    # fpr = False Positive Rate
    # tpr = True Posisive Rate 
    # th = probability thresholds
                                
AUC = auc(fpr, tpr) # Calculating Area Under Curve using sklearn function 
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('ROC Curve - Living Status based on Albumin Levels')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate (FPR)')
plt.ylabel('TRUE Positive Rate (TPR)')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC) # AUC value shown in legend 
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # Dotted Linear reference line 
                                                        
plt.legend(loc=LL)
plt.show()
###############

"""
 4.) Add comments to explain the code blocks.

    Code Has Been Added to the Code Blocks Above.
"""

######
   
"""
  5.) Add a summary comment block that discusses your classification and its accuracy.
    
    
I utilized Naive Bayes classifier to ask the question whether a Hepatitis patient's albumin
levels influence whether a patient lives or dies. The classifier was passed a 
probability threshold of 0.6, and that has lead to a precision score of 0.71,
a recall score for 0.63, and F1 score of 0.67. These accuracy metrics show that the 
classifer is reliable and accurate. 

In addition, the True positive rate (TPR) is 0.92, and the False positive rate (FPR) is 0.37. 

Overall, I believe the classification did a good job.  That AUC was 0.77, which is good. 
The ROC curve could be a little bit better, as it could be a little more closer 
to the upper left - but it is not bad. 



"""

