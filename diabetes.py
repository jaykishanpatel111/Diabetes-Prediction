#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


# In[2]:


cd F:\Dataset\Done projects\diabetes data set


# In[3]:


pwd


# # Read Dataset

# In[4]:


df = pd.read_csv('diabetes2.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


col = df.columns
col


# In[8]:


df.describe()


# # Checking null values in the dataset

# In[9]:


df.isna().sum()


# # Checking Duplicate in data

# In[10]:


df.duplicated().sum()


# In[11]:


df.drop_duplicates(inplace = True)


# In[12]:


df.shape


# # Plot Histogram mapes

# In[13]:


fig , s= plt.subplots(3,2, figsize = (15,10))
s[0][0].set_title("Histogram of pregnancies column")
s[1][0].set_title("Histogram of Glucose column")
s[2][0].set_title("Histogram of BloodPressure column")
s[0][1].set_title("Histogram of Insulin column")
s[1][1].set_title("Histogram of SkinThickness column")
s[2][1].set_title("Histogram of BMI column")

s[0][0].hist(df['Pregnancies'])
s[1][0].hist(df['Glucose'])
s[2][0].hist(df['BloodPressure'])
s[0][1].hist(df['Insulin'] )
s[1][1].hist(df['SkinThickness'])
s[2][1].hist(df['BMI'])
plt.show()


# # Plot Scatterplot

# In[14]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'Age',y= 'Pregnancies', hue = 'Outcome', data = df)
plt.show()


# In[15]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'Age',y= 'Glucose', hue = 'Outcome', data = df)
plt.show()


# In[16]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'Age',y= 'BloodPressure', hue = 'Outcome', data = df)
plt.show()


# In[17]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'Age',y= 'SkinThickness', hue = 'Outcome', data = df)
plt.show()


# In[18]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'Age',y= 'Insulin', hue = 'Outcome', data = df)
plt.show()


# In[19]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'Age',y= 'BMI', hue = 'Outcome', data = df)
plt.show()


# In[20]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'Age',y= 'DiabetesPedigreeFunction', hue = 'Outcome', data = df)
plt.show()


# # Filling missing values

# In[21]:


df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(0, df['DiabetesPedigreeFunction'].mean())

df.head()


# # Correlation between each features

# In[22]:


corr_data = df.corr()


# In[23]:


corr_data


# # Ploting heat map of the correlated data

# In[24]:


plt.figure(figsize = (10,5))
sns.heatmap(corr_data, annot = True, cmap = 'RdYlGn')


# # Scaling data

# In[25]:


from sklearn.preprocessing import RobustScaler
scaling = RobustScaler()
df = scaling.fit_transform(df)


# In[26]:


df = pd.DataFrame(df,columns=col)
df.head()


# # Model Building

# In[27]:


# Create features and target data


# In[28]:


X = df.drop(['Outcome'], axis= 1)
Y = df.Outcome


# In[29]:


X.shape, Y.shape


# # Spliting training and testing dataset

# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30,random_state=11)


# In[31]:


X_train.shape, Y_train.shape  , X_test.shape, Y_test.shape


# # Creating Random Forest model

# In[32]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 15, random_state = 11)


# In[33]:


# train the model
model.fit( X_train , Y_train.ravel())


# In[34]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score


# In[35]:


# Predicting values from the model
Y_pred = model.predict(X_test)
Y_pred = np.array([0 if i < 0.5 else 1 for i in Y_pred])
Y_pred


# # Checking accuracy score of our model

# In[36]:


def run_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train.ravel())
    accuracy = accuracy_score(Y_test, Y_pred)
    print("pricison_score: ",precision_score(Y_test, Y_pred))
    print("recall_score: ",recall_score(Y_test, Y_pred))
    print("Accuracy = {}".format(accuracy))
    print(classification_report(Y_test,Y_pred,digits=5))
    print(confusion_matrix(Y_test,Y_pred))


# In[37]:


run_model(model, X_train, Y_train, X_test, Y_test)


# In[38]:


cm = confusion_matrix(Y_test, Y_pred)
cm


# In[39]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# # Classification Report

# In[40]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# In[ ]:




