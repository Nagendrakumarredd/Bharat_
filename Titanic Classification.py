#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set the random seed state for reproducibility
random_state = 42


# In[3]:


train_df = pd.read_csv("C:\\Users\\syama\\Desktop\\train.csv")
test_df = pd.read_csv("C:\\Users\\syama\\Desktop\\test.csv")


# In[4]:


train_df.head()


# In[5]:


test_df.head()


# In[6]:


train_df.info()


# In[7]:


id_variable= 'PassengerId'
dependent_variable = 'Survived'


# In[8]:


sns.catplot(x=dependent_variable, data=train_df, kind='count').set(title=dependent_variable)
plt.show()


# In[9]:


sns.catplot(x='Pclass', hue=dependent_variable, data=train_df, kind='count').set(title='Pclass and Survived')
plt.show()


# In[10]:


sns.catplot(x='Sex', hue=dependent_variable, data=train_df, kind='count').set(title='Sex and Survived')
plt.show()


# In[11]:


train_df.describe()


# In[13]:


train_df['Cabin'] = train_df['Cabin'].fillna('NA') 
test_df['Cabin'] = test_df['Cabin'].fillna('NA')


# In[14]:


train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])


# In[15]:


train_age_mean = train_df['Age'].mean()

train_df['Age'] = train_df['Age'].fillna(train_age_mean)

test_df['Age'] = test_df['Age'].fillna(train_age_mean)


# In[16]:


train_fare_median = train_df['Fare'].median()

test_df['Fare'] = test_df['Fare'].fillna(train_fare_median)


# In[17]:


train_df.info()


# In[18]:


age_bins = [0, 18, 65, 100]
age_labels = ['child','adult', 'elderly']

train_df['age_bucket'] = pd.cut(x=train_df['Age'], bins=age_bins,labels=age_labels).astype('object')

test_df['age_bucket'] = pd.cut(x=test_df['Age'], bins=age_bins, labels=age_labels).astype('object')


# In[19]:


sns.catplot(x='age_bucket', hue=dependent_variable, data=train_df, kind='count').set(title='age_bucket and Survived')
plt.show()


# In[22]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[23]:


y_train = train_df[dependent_variable].copy()
x_train = train_df.drop(columns=[id_variable,dependent_variable], axis=1).copy()


# In[25]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[26]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# In[27]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# In[28]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[29]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[ ]:





# In[30]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[34]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[36]:


train_df['Age'] = train_df[['Age','Pclass']].apply(impute_age,axis=1)


# In[37]:


train_df['Embarked'] = train_df['Embarked'].fillna('S')


# In[40]:


sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[42]:


train_df.drop('Cabin',axis=1,inplace=True)


# In[43]:


train_df.head()


# In[44]:


train_df.dropna(inplace=True)


# In[46]:


sex = pd.get_dummies(train_df['Sex'],drop_first=True)
embark = pd.get_dummies(train_df['Embarked'],drop_first=True)


# In[53]:


train_df.drop(['age_bucket'],axis=1,inplace=True)


# In[54]:


train_df = pd.concat([train_df,sex,embark],axis=1)


# In[55]:


train_df.head()


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Survived'],axis=1), 
                                                    train['Survived'], test_size=0.10, 
                                                    random_state=101)


# In[ ]:





# In[58]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[59]:


predictions = logmodel.predict(X_test)
X_test.head()


# In[60]:


predictions


# In[61]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[63]:


print(classification_report(y_test,predictions))


# By using the Logistic Regression as the classification method , I got the accuracy as the 80%. 
# 

# In[ ]:




