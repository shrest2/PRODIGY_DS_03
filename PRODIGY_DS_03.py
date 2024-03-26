#!/usr/bin/env python
# coding: utf-8

# ## Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.
# 

# In[28]:


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[29]:


# Load the dataset

df=pd.read_csv("C:/Users/HP/Desktop/Internship/Prodigy/bank-dataset.csv")  # Check the delimiter; it might be different


# In[30]:


df.head(10)


# In[31]:


# Subsetting numeric columns
numeric_df = df.select_dtypes(include=['float', 'int'])

# Boxplots for numeric data
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
ax = ax.flatten()
numeric_col_iter = iter(numeric_df.columns.to_list())

for axis in list(ax):
    try:
        column_name = next(numeric_col_iter)
        sns.boxplot(data=numeric_df[column_name], ax=axis, orient='h')
        axis.set_title(f"Boxplot of {column_name.title()}")

    except StopIteration:
        axis.remove()

plt.tight_layout()


# In[32]:


plt.tight_layout()


# In[33]:


#Distribution of Age
sns.distplot(df.age, bins=100,color="red")


# In[36]:


#Visualizing the distribution of the 'duration' column
sns.distplot(df.duration, bins=100,color="yellow")


# In[38]:


#Copy for parsing
bank_data = df.copy()


# In[39]:


#Exploring People who made a deposit Vs Job category
jobs = ['management','blue-collar','technician','admin.','services','retired','self-employed','student',        'unemployed','entrepreneur','housemaid','unknown']

for j in jobs:
    print("{:15} : {:5}". format(j, len(bank_data[(bank_data.deposit == "yes") & (bank_data.job ==j)])))


# In[46]:


#Converting categorical variables to dummies
bank_with_dummies = pd.get_dummies(data=bank_data, columns = ['job', 'marital', 'education', 'poutcome'],                                    prefix = ['job', 'marital', 'education', 'poutcome'])
bank_with_dummies.head()


# In[41]:


#Scatterplot showing age and balance
bank_with_dummies.plot(kind='scatter', x='age', y='balance');


# In[50]:


bank_with_dummies[bank_with_dummies.marital_married == 1].describe()


# In[51]:


#Copying
bankcl = bank_with_dummies


# In[55]:


corr = bankcl.corr()
corr


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train-Test split: 20% test data
data_drop_deposite = bankcl.drop('marital_married', axis=1)

label = bankcl.marital_married
data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size=0.2, random_state=50)

# Decision tree with depth = 2
dt2 = DecisionTreeClassifier(random_state=1, max_depth=2)
dt2.fit(data_train, label_train)

# Plotting the decision tree
plt.figure(figsize=(10, 8))
plot_tree(dt2,
          feature_names=data_drop_deposite.columns.tolist(),
          class_names=['Not Signed Up', 'Signed Up'],
          filled=True,  # Fill nodes with colors
          rounded=True,  # Round the boxes
          fontsize=10,  # Font size for the text in the tree
          )
plt.show()


# In[ ]:




