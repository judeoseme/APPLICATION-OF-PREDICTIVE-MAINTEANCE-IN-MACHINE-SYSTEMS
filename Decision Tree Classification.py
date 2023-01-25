#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# loading dataset
dataset = pd.read_csv('predictive_main.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.head()


# In[5]:


dataset.info()


# In[7]:


dataset.describe()


# In[8]:


dataset.describe(include = 'all')


# In[9]:


sns.histplot(dataset.Type)
plt.title('Type distribution')
plt.show()


# In[10]:


plt.figure(figsize=(10, 5))
plt.title('machinefailure based on the type')
sns.histplot(x='Type', hue='Machinefailure', data = dataset)
plt.show()


# In[11]:


dataset['TWF']=dataset['TWF'].replace(['M','L',],[0, 1])


# In[12]:


dataset.head()


# In[30]:


x = dataset.iloc[:, [3, 4, 5, 6, 7, 9, 10, 11, 12,13 ]].values
y = dataset.iloc[:, 8].values  


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state =0)


# In[32]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_s=sc.fit_transform(x_train)
x_test_s=sc.transform(x_test)


# In[33]:


# fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
Classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Classifier.fit(x_train, y_train)


# In[34]:


# predicting the Test set results
y_pred=Classifier.predict(x_test_s)
print(y_pred)


# In[35]:


print(y_test)


# In[36]:


# to evalaute the performance model
from sklearn import metrics
acc = metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(cm, '\n\n')
print('-------------------------------------------------')
result = metrics.classification_report(y_test, y_pred)
print('Classification Report:\n')
print(result)


# In[37]:


ax = sns.heatmap(cm, cmap='flare', annot= True, fmt = 'd')

plt.xlabel('Predicted Class', fontsize = 12)
plt.ylabel('True Class', fontsize = 12)
plt.title('Confusion Matrix', fontsize = 12)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




