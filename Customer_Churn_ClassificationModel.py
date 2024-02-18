#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries-->

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Importing the dataset and checking the first five rows-->

df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()


# In[3]:


#Checking general information of our dataset-->

df.info()


# In[4]:


#Checking the statistical values-->

df.describe().T

#From above we can conclude->

*only 25% of population is having tenure more than 55 months, rest of them are having less than 55
*only 25% of population is paying more than $90 per month
# In[5]:


#Checking the null values-->

df.isnull().sum()

#Our data has no null values in any of the column
# In[6]:


#Checking the percentage of each category in our target column-->

df['Churn'].value_counts(normalize=True)*100

#Observations->

*Almost 27% of our populatuion is churning
*Our data seems to be highly imbalanced
# In[7]:


#Plotting a count plot for our target variable by keeping 'gender' on hue parameter-->

sns.countplot(x='Churn', data=df, hue='gender')
plt.show()

#From above we can conclude->

*Gender doesn't have a huge role to play in churning of the customer
# # Exploratory Data Analysis-->

# In[8]:


#Converting 'TotalCharges' into numeric column-->

df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
df.info()


# In[9]:


df.describe()


# In[10]:


#Imputing the missing values in 'TotalCharges' with median value-->

df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.isnull().sum()


# In[11]:


#Dropping the irrelevant column 'customerID'-->

df.drop(columns=['customerID'], inplace=True)


# In[12]:


#Creating numcol for all numeric dtypes columns-->

numcol=df.select_dtypes(include='number')
numcol.columns


# In[13]:


#Creating catcol for all object dtypes columns-->

catcol=df.select_dtypes(include='object')
catcol.columns


# # Univariate Analysis-->

# In[14]:


#Creating a frequency plot for 'TotalCharges', 'MonthlyCharges' & 'tenure'-->

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.kdeplot(x='TotalCharges',data=df, color='red', hue='Churn')

plt.subplot(2,2,2)
sns.kdeplot(x='MonthlyCharges', data=df,color='blue', hue='Churn')

plt.subplot(2,2,3)
sns.kdeplot(x='tenure', data=df, color='white', hue='Churn')

plt.show()

#Observations->

          *As the monthly charges increases, the numn of churners are also increasing
          *The num of churners are at higher end at lower price in total charges which is unexpected
          *As the tenure increases, the num of churners decreases 
# In[15]:


#Creating a countplot for catcol by keeping 'Churn' into hue parameter-->

for i in catcol:
    plt.figure(figsize=(6,4))
    sns.countplot(x=i, data=df, hue='Churn')
    plt.xticks(rotation=45)
    plt.show()

#Insights from above->

            *Gender doesn't have a role to play in churning and so does 'Multilines'.
            *Customers with 'Fiber Optic' & 'Phone Service' and with no 'Partner' & no 'Dependents' are high churners.
            *Also customers with no 'Online Security', no 'Online Backup', no 'Device Protection', no 'Tech Support', no                      'Streaming TV' & no 'Streaming Movies' are high churners.
            *Customers with 'Month-to-Month' contract, 'Paperless billing', 'Electronic checks' as their payment method have                  been churning more.
# In[16]:


#Plotting a Heatmap to check the correlation of numeric variables-->

sns.heatmap(df.corr(), annot=True)
plt.show()

Observation->
         *'Total Charges' is highly correlated to 'Tenure'
# In[17]:


#Creating a Boxplot for numcol-->

for i in numcol:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=i, data=numcol)
    plt.xticks(rotation=45)
    plt.show()

#Observation->
           *There are no Outliers in our dataset
# # Bivariate Analysis-->

# In[18]:


#Creating a pairplot-->

sns.pairplot(df)
plt.show()


# # Grouping our data-->

# In[19]:


#Using groupby function to group 'churn' with the mean of 'TotalCharges' ,'tenure' & 'MonthlyCharges'-->


df.groupby('Churn')[['TotalCharges','MonthlyCharges','tenure']].agg('mean')

#From above, we can conclude-->

  *Customers with higher 'Monthly Charges' are the ones who churns more whereas customers with less 'Total charges' churns more
  *The avg tenure of churning customers is almost 18 months
# # Encoding-->

# In[20]:


#Converting categorical features into numeric features-->

df1=pd.get_dummies(data=df, columns=['gender','SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod'], drop_first=True)
df1.head(2)


# # Splitting our dataset into Feature set & DV-->

# In[21]:


#Creating Feature set as 'x' & DV as 'y'-->

x=df1.drop(columns='Churn')
y=df1['Churn']


# In[22]:


#Importing train_test_split module-->

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.25, random_state=111)


# # Feature Scaling-->

# In[23]:


#Normalizing our Training dataset-->

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# # Model Fitting-->

# ## Logistic Regression-

# In[24]:


#Logistic Regression Model-->

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
logr.fit(x_train,y_train)


# In[25]:


#Making predictions-->

logr_pred=logr.predict(x_test)


# In[26]:


#Checking the accuracy of our model-->

from sklearn.metrics import confusion_matrix, classification_report
logr_report=classification_report(y_test, logr_pred)
print('classification report->','\n',logr_report)
logr_matrix=confusion_matrix(y_test, logr_pred)
print('confusion matrix->','\n',logr_matrix)

#So the accuracy of our Logistic Regression model is coming to be 81%
# ## Random Forest Classifier-

# In[27]:


#Random Forest Classifier-->

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train, y_train)


# In[28]:


#Making predictions-->

rfc_pred=rfc.predict(x_test)


# In[29]:


#Checking the accuracy of our model-->

rfc_report=classification_report(y_test, rfc_pred)
print('classification report->','\n',rfc_report)
rfc_matrix=confusion_matrix(y_test, rfc_pred)
print('confusion matrix->','\n',rfc_matrix)

#So the accuracy of our Random Forest Classifier is coming to be 79%
# ## Support Vector Classifier-

# In[30]:


#Support Vector Classifier-->

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)


# In[31]:


#Making predictions-->

svc_pred=svc.predict(x_test)


# In[32]:


#Checking the accuracy of our model-->

svc_report=classification_report(y_test, svc_pred)
print('classification report->','\n',svc_report)
svc_matrix=confusion_matrix(y_test, svc_pred)
print('confusion matrix->','\n',svc_matrix)

#So the accuracy of our Support Vector Classifier is coming to be 80%
# ## KNearestNeighbors Model-

# In[33]:


#KNearestNeighbors Model-->

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)


# In[34]:


#Making predictions-->

knn_pred=rfc.predict(x_test)


# In[35]:


#Checking the accuracy of our model-->

knn_report=classification_report(y_test, knn_pred)
print('classification report->','\n',knn_report)
knn_matrix=confusion_matrix(y_test, knn_pred)
print('confusion matrix->','\n',knn_matrix)

#So the accuracy of our Support Vector Classifier is coming to be 79%#Conclusion->
         *Out of all the models, Logistic Regression has the highest accuracy of 81%
# In[ ]:




