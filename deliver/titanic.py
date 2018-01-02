
# coding: utf-8

# In[1]:


import urllib.request
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


deliver = os.path.abspath(os.path.dirname('__file__'))
data_dir = os.path.abspath(os.path.relpath('../input/', deliver))


# ## Import into pandas

# In[17]:


df = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
df.tail()


# In[18]:


df = pd.concat([df, pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col=0)], axis=0)
df.head()


# ## Age

# In[19]:


df.Age.hist()
df[df.Survived == 1].Age.hist()
plt.xlabel("Age", fontsize=16)
plt.legend(["total", "survived"], fontsize=14)
plt.show()


# ### Scaling

# In[20]:


age_delta = df.Age.max() - df.Age.min()
df["Age"] = (df.Age - df.Age.min())/age_delta


# ### Missing values to average

# In[25]:


df.loc[df.Age.isnull()]["Age"] = df.Age.mean()


# In[26]:


df.Age.hist()
df[df.Survived == 1].Age.hist()
plt.xlabel("Age", fontsize=16)
plt.legend(["total", "survived"], fontsize=14)
plt.show()


# ## Sex

# In[27]:


df.loc[df.Sex == "female"]["Sex"] = 1
df.loc[df.Sex == "male"]["Sex"] = 0


# In[28]:


df.Sex.hist()
df[df.Survived == 1].Sex.hist()
plt.xlabel("Sex", fontsize=16)
plt.legend(["total", "survived"], fontsize=14)
plt.show()


# In[ ]:




