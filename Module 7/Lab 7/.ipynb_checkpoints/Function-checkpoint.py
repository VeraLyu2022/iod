#!/usr/bin/env python
# coding: utf-8

# ## libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


def visualization_box_dist(df):
    for c in df.columns:
        fig, ax = plt.subplots(2,
                               figsize = (10, 5),
                               sharex = True,
                               gridspec_kw = {'height_ratios': (0.15, 0.85)})
    
        ax_box = ax[0]
        ax_box = sns.boxplot(df[c], ax = ax_box)
        ax_box.set(title = c, yticks = [], xlabel = '')
        sns.despine(ax = ax_box, left = True)
    
        ax_hist = ax[1]
    
        if c == 'Donated_Mar_2007':
            ax_hist = sns.distplot(df[c], kde=False, ax = ax_hist, orient='h')
            ax_hist.set(xlabel = '')
            sns.despine(ax = ax_hist)
        else:
            ax_hist = sns.distplot(df[c], ax = ax_hist)
            ax_hist.set(xlabel = '')
            sns.despine(ax = ax_hist)
    
    
    plt.show()

