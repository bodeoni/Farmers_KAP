#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data wrangling and visualisations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Stats
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import scikit_posthocs as sp
from scipy.stats import kstest as KS
from scipy.stats import shapiro

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Check for versions

import scipy
import sys

print(sys.version)
print('Pandas: ', pd.__version__)
print('Numpy: ', np.__version__)
print('Scikit_Posthocs: ', sp.__version__)
print('Scipy: ', scipy.__version__)


# In[3]:


#Read in the data
data = pd.read_excel('Farmers training coding book.xlsx', na_values= 99)


# In[4]:


#Check Read
data.head()


# In[5]:


#Drop perception column
data.drop('Perception', axis=1, inplace=True)


# ### Farmer (Participants) Characterisitcs

# In[6]:


#Percentage of responses from different trainings
print(data['VA'].value_counts())
print(data['VA'].value_counts(normalize=True) * 100)


# In[7]:


#Sumamry statistics for some farmer characteristcs
cols =['V1','V3','V4', 'V5', 'V6', 'V7','V28']
labels=['Farmers Asscoaiton', 'Extesnsion Visit', 'Age Group', 'Gender', 'Farming Practice', 'Previous Training',
       'Multi-Cropping']
n=0
for i in cols:
    a=data[i].value_counts()
    b=data[i].value_counts(normalize=True)
    print(labels[n])
    print(a)
    print(b)
    print('---')
    n=n+1


# In[8]:


# How long have participants being farming cassava
data['V30'].describe()


# In[9]:


data['V30'].agg('std')


# In[10]:


# Of the commercial farmers, how do they sell
print('Processed')
print(data['V22_1'].value_counts())
print(data['V22_1'].value_counts(normalize=True))
print('---')
print('Tuber')
print(data['V22_2'].value_counts())
print(data['V22_2'].value_counts(normalize=True))


# In[11]:


# How long have study participants been growing cassava
data['V30'].describe()


# In[12]:


sns.boxplot(data['V30'])


# In[13]:


#Years of practice by age group
data[['V4','V30']].groupby('V4')['V30'].agg(['count','mean', 'std'])


# In[14]:


# What were the tubers most frequently processed into?
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[15]:


# Recast column to string data type
data['V23']= data['V23'].astype('str')


# In[16]:


stopwords = set(STOPWORDS)
stopwords.update(['nan'])


# In[17]:


#generate wordcloud 
wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(' '.join(data['V23']))


# In[18]:


plt.imshow(wordcloud, interpolation='bilinear')
plt.title('What participants process cassava into')


# ### a quick look at farmers who multicrop

# In[19]:


print('Multicropping')
print(data['V28'].value_counts())
print(data['V28'].value_counts(normalize=True))


# In[20]:


# Recast column to string data type
data['V29']= data['V29'].astype('str')

#generate wordcloud 
wordcloud2 = WordCloud(stopwords=stopwords, background_color='white').generate(' '.join(data['V29']))
plt.title('Crops planted alongside Cassava')

plt.imshow(wordcloud2, interpolation='bilinear')


# ## Do knowledge and practice scores come from a normal distribution?

# In[21]:


# Check if Kowledge and Practice come from a normal distribution
# Shapiro-Wilk
print('Shapiro - Wilk Test')
print('---')
print('Knowledge: ', shapiro(data['Knowledge']))
print('Practice: ', shapiro(data['Practice']))


print()
#Kolmogorov-Smimov Test
print('Kolmogorov-Smimov Test')
print('---')
print('Knowledge: ', KS(data['Knowledge'], 'norm'))
print('Practice: ', KS(data['Practice'], 'norm'))


# Its safe to conclude that Knowledge and Practice Scores deviate from the normal

# ---

# # Knowledge

# In[22]:


data['Knowledge'].describe()


# ### Do people who have seen the symptoms in their farms before know more than others?

# In[23]:


print(data[['V8', 'Knowledge']].groupby('V8')['Knowledge'].agg(['mean','median','min','max']))
print(data[['V10', 'Knowledge']].groupby('V10')['Knowledge'].agg(['mean','median','min','max']))


# In[24]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.boxplot(data=data, x= data['Knowledge'], y=data['V8'])
plt.title('Have you seen Image 1 on your farm?')

plt.subplot(1,2,2)
sns.boxplot(data=data, x= data['Knowledge'], y=data['V10'])
plt.title('Have you seen Image 2 on your farm?')

plt.tight_layout()


# In[25]:


#Create dummy varaibles from variables measuring if farmer has seen symptom
data['V8_dummy'] = data['V8'].apply(lambda x: 1 if x == 'Yes' else 0)
data['V10_dummy'] = data['V10'].apply(lambda x: 1 if x == 'Yes' else 0)

#add both columns together to determine how many of the symptoms the farmer has seen
data['Symptom'] = data['V8_dummy'] + data['V10_dummy']


# In[26]:


sns.boxplot(data=data, x= data['Knowledge'], y=data['Symptom'].astype('category'))
plt.title('Number of symptoms seen')


# In[27]:


# descriptives for number of symptoms seen
data[['Knowledge', 'Symptom']].groupby('Symptom')['Knowledge'].agg(['count','mean','median','min','max'])


# In[28]:


# percentage of persons in relation to how many symptoms they have seen
data['Symptom'].value_counts(normalize=True)


# In[29]:


# Does Knowledge score differ between persons who have see 1 vs 2 symptoms
stat, p = mannwhitneyu(data.loc[data['Symptom'] == 1, 'Knowledge'], 
                       data.loc[data['Symptom'] == 2, 'Knowledge']
                      )

print('Statistics=%.3f \n p=%.4f' % (stat, p))


# #### it would seem that having seen the symptoms on one's farm is predictive of knowledge

# In[30]:


plt.figure(figsize=(15,8))

plt.subplot(1,3,1)
sns.distplot(data.loc[data['Symptom'] == 2, 'Knowledge'])
plt.title('Seen both Symptoms')

plt.subplot(1,3,2)
sns.distplot(data.loc[data['Symptom'] == 1, 'Knowledge'])
plt.title('Seen one Symptom')

plt.subplot(1,3,3)
sns.distplot(data.loc[data['Symptom'] == 0, 'Knowledge'])
plt.title('Seen no Symptom')


# In[31]:


#Create column to check if participant has seen at least one symptom
data['Seen_One'] = data['Symptom'].apply(lambda x: 'Yes' if x >= 1 else 'No')


# In[32]:


data[['Seen_One', 'Knowledge']].groupby('Seen_One')['Knowledge'].agg(['count', 'mean','std','median','min','max'])


# In[33]:


#Percentage of those who have seen at least one of the symptoms
data['Seen_One'].value_counts(normalize=True)*100


# In[34]:


sns.boxplot(data=data, x='Seen_One', y= 'Knowledge')


# In[35]:


# Are farmers who have seen at least one symptom in their farm more likely to know more
stat, p = mannwhitneyu(data.loc[data['Seen_One'] == 'Yes', 'Knowledge'], 
                       data.loc[data['Seen_One'] == 'No', 'Knowledge'], 
                       alternative='two-sided')

print('Statistics=%.3f \n p=%.4f' % (stat, p))


# In[36]:


# Is there a significant difference in the knowledge scores based on how many symptoms have been seen
stat, p = kruskal(data.loc[data['Symptom'] == 0, 'Knowledge'], 
                  data.loc[data['Symptom'] == 1, 'Knowledge'], 
                  data.loc[data['Symptom'] == 2, 'Knowledge']
                  )

print('Statistics=%.3f \n p=%.4f' % (stat, p))


# In[37]:


x= [data.loc[data['Symptom'] == 0, 'Knowledge'], 
    data.loc[data['Symptom'] == 1, 'Knowledge'], 
    data.loc[data['Symptom'] == 2, 'Knowledge']]


# In[38]:


#post hoc with Conover test
pc = sp.posthoc_conover(x)
heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(pc, **heatmap_args)


# In[39]:


# Post hoc with mann whitney
pc2 = sp.posthoc_mannwhitney(x)
heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(pc2, **heatmap_args)


# In[40]:


# Create knowledge bins
bins= [0,3,6,8]
labels= ['Low', 'Mid', 'High']
data['Knowledge_bin'] = pd.cut(data['Knowledge'], bins=bins, labels= labels )


# In[41]:


print(data['Knowledge_bin'].value_counts())
print(data['Knowledge_bin'].value_counts(normalize=True)*100)


# In[42]:


# What is the common perception about what causes mosaic and leaf distortion
cols = ['V9_1', 'V9_2', 'V9_3','V9_4','V9_5']
labels= ['Rain', 'Variety', 'Age', 'Virus', 'No Idea']
c=0
for i in cols:
    a= data[i].value_counts()
    b= data[i].value_counts(normalize=True)
    print(labels[c])
    print(a)
    print(b)
    print('')
    c=c+1


# In[43]:


# What is the common perception about what causes rottening of the tubers
cols = ['V11_1', 'V11_2', 'V11_3','V11_4','V11_5']
labels= ['Rain', 'Variety', 'Age', 'Virus', 'No Idea']
c=0
for i in cols:
    a= data[i].value_counts()
    b= data[i].value_counts(normalize=True)
    print(labels[c])
    print(a)
    print(b)
    print('')
    c=c+1


# In[44]:


# Descriptives for some other knowledge questions
cols = ['V15', 'V16', 'V17','V18']
labels= ['Whitefly', 'Chemical Prevention', 'Burning', 'Rain']
c=0
for i in cols:
    a= data[i].value_counts()
    b= data[i].value_counts(normalize=True)
    print(labels[c])
    print(a)
    print(b)
    print('')
    c=c+1


# In[45]:


data[['Knowledge','V30']].corr(method='spearman')


# ---

# # Practice

# In[46]:


#Descriptive Stats for practice scores
data['Practice'].describe()


# In[47]:


#Distribution of practice scores
sns.kdeplot(data['Practice'])


# In[48]:


#How many farmers use chemical to prevent CMD
print(data['V31'].value_counts())
print(data['V31'].value_counts(normalize=True) *100)


# In[49]:


#Clean up the wrongly coded value
data['V31'].replace(9, np.nan, inplace=True)


# In[50]:


#How many farmers use chemical to prevent CMD
print(data['V31'].value_counts())
print(data['V31'].value_counts(normalize=True) *100)


# In[51]:


#How many remove plants showing symptoms
data['V32'].value_counts()


# In[52]:


#Clean up the wrongly coded value
data['V32'].replace(9, np.nan, inplace=True)


# In[53]:


#How many remove plants showing symptoms
print(data['V32'].value_counts())
print(data['V32'].value_counts(normalize=True) *100)


# In[54]:


#of those who remove them, are they burnt
print(data['V33'][data['V32']=='Yes'].value_counts())
print(data['V33'][data['V32']=='Yes'].value_counts(normalize=True)*100)


# In[55]:


#Do persons who remove infected plants have better knowledge
data[['V32','Knowledge']].groupby('V32')['Knowledge'].agg(['count','mean','std','median','min','max'])


# In[56]:


#Does knowledge differ based on whether or not a person removes infected plant
mannwhitneyu(data.loc[data['V32'] == 'Yes', 'Knowledge'],
             data.loc[data['V32'] == 'Yes', 'Knowledge'])


# In[57]:


cols=['V34_1','V34_2','V34_3']
headers=['Reputable Source', 'Previous Planting Season', 'From Neighbouring farms']
j=0

for i in cols:
    print(headers[j])
    print(data[i].value_counts())
    print(data[i].value_counts(normalize=True)*100)
    print(" ")
    j=j+1


# In[58]:


#How many farmers had obtained cuttings from otuside their state
print(data['V35'].value_counts())
print(data['V35'].value_counts(normalize=True)*100)


# In[59]:


data['V36'].value_counts()


# In[60]:


#How many farmers had obtained cuttings from otuside their country
print(data['V37'].value_counts())
print(data['V37'].value_counts(normalize=True)*100)


# In[61]:


data['V38'].value_counts()


# #### Number of symptoms seen Vs Practice

# In[62]:


data[['Symptom', 'Practice']].groupby('Symptom')['Practice'].agg(['count','mean','std','median','min','max'])


# In[63]:


data[['Seen_One', 'Practice']].groupby('Seen_One')['Practice'].agg(['count','mean','std','median','min','max'])


# In[64]:


#does seeing a symptoms previously affect practice scores
mannwhitneyu(data.loc[data['Seen_One']=='Yes', 'Practice'],
            data.loc[data['Seen_One']=='No', 'Practice'])


# In[65]:


kruskal(data.loc[data['Symptom']== 0, 'Practice'],
       data.loc[data['Symptom']== 1, 'Practice'],
       data.loc[data['Symptom']== 2, 'Practice'])


# ---

# # Knowledge/Practice and its Covariates

# ### Does belonging to a farmers association affect knowlege and practice?

# In[66]:


#How many in each category
data['V1'].value_counts()


# In[67]:


data[['V1', 'Knowledge', 'Practice']].groupby('V1')['Knowledge','Practice'].agg(['count','mean','std'])


# In[68]:


axis_font = {'fontname':'Arial', 'size':'15'}
plt.subplots(1, 2, figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=data, x= 'Knowledge', y='V1',orient='h')
plt.xlabel('Knowledge', **axis_font)
plt.ylabel('Belong to a Farmers Association', **axis_font)

plt.subplot(1,2,2)
sns.boxplot(data=data, x= 'Practice', y='V1', orient='h')
plt.xlabel('Practice', **axis_font)
plt.ylabel('Belong to a Farmers Association', **axis_font)

plt.tight_layout()


# In[69]:


#Knowledge vs Farmers Assocaiton
stat, p = mannwhitneyu(data.loc[data['V1'] == 'Yes', 'Knowledge'], 
                       data.loc[data['V1'] == 'No', 'Knowledge'], 
                       alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))

#Practice vs Farmers Association
stat, p = mannwhitneyu(data.loc[data['V1'] == 'Yes', 'Practice'], 
                       data.loc[data['V1'] == 'No', 'Practice'],
                      alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))


# #### Belonging to a farmers association did not siginificantly increase farmer knowledge or practice

# ***

# ### How does previous extension officer visits affect dependent variables

# In[70]:


#How many in each category
data['V3'].value_counts()


# In[71]:


axis_font = {'fontname':'Arial', 'size':'15'}
plt.subplots(1, 2, figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=data, x= 'Knowledge', y='V3',orient='h')
plt.xlabel('Knowledge', **axis_font)
plt.ylabel('Extension officer visit', **axis_font)

plt.subplot(1,2,2)
sns.boxplot(data=data, x= 'Practice', y='V3', orient='h')
plt.xlabel('Practice', **axis_font)
plt.ylabel('Extension officer visit', **axis_font)

plt.tight_layout()


# In[72]:


data[['V3', 'Knowledge', 'Practice']].groupby('V3')['Knowledge','Practice'].agg(['count','mean','std'])


# In[73]:


#Knowledge vs Extension Visit
stat, p = mannwhitneyu(data.loc[data['V3'] == 'Yes', 'Knowledge'], 
                       data.loc[data['V3'] == 'No', 'Knowledge'], 
                       alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))

#Practice vs Extesnion Visit
stat, p = mannwhitneyu(data.loc[data['V3'] == 'Yes', 'Practice'], 
                       data.loc[data['V3'] == 'No', 'Practice'],
                      alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))


# #### It would seem that persons who had been visited by extension officers were more likely to have better practice than those who had not. The same however cannot be said about knowledge

# ***

# ### Scores by Age Group

# In[74]:


data[['V4', 'Knowledge', 'Practice']].groupby('V4')['Knowledge','Practice'].agg(['count','mean','std'])


# In[75]:


stat, p = kruskal(data.loc[data['V4'] == '31-40', 'Knowledge'], 
                  data.loc[data['V4'] == '41-50', 'Knowledge'],
                  data.loc[data['V4'] == '51-60', 'Knowledge'],
                  data.loc[data['V4'] == '61 and above', 'Knowledge'],
                  data.loc[data['V4'] == '20-30', 'Knowledge'],
                  data.loc[data['V4'] == 'below 20', 'Knowledge'], 
                )
print('Statistics=%.3f \n p=%.3f' % (stat, p))

stat, p = kruskal(data.loc[data['V4'] == '31-40', 'Practice'], 
                  data.loc[data['V4'] == '41-50', 'Practice'],
                  data.loc[data['V4'] == '51-60', 'Practice'],
                  data.loc[data['V4'] == '61 and above', 'Practice'],
                  data.loc[data['V4'] == '20-30', 'Practice'],
                  data.loc[data['V4'] == 'below 20', 'Practice'],
                 )
print('Statistics=%.3f \n p=%.3f' % (stat, p))


# In[76]:


order = ['below 20', '20-30','31-40','41-50','51-60','61 and above']
axis_font = {'fontname':'Arial', 'size':'15'}
plt.subplots(1, 2, figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=data, x= 'Knowledge', y='V4',orient='h', order=order)
plt.xlabel('Knowledge', **axis_font)
plt.ylabel('Age Group', **axis_font)

plt.subplot(1,2,2)
sns.boxplot(data=data, x= 'Practice', y='V4', orient='h', order=order)
plt.xlabel('Practice', **axis_font)
plt.ylabel('Age Group', **axis_font)

plt.tight_layout()


# ***

# ### Gender versus dependent Variables

# In[77]:


# Number of persons by sex
data['V5'].value_counts()


# In[78]:


#Average practice and knowledge score for each sex
data[['V5','Knowledge','Practice']].groupby('V5')['Knowledge', 'Practice'].agg(['count','mean','std'])


# In[79]:


axis_font = {'fontname':'Arial', 'size':'15'}
plt.subplots(1, 2, figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=data, x= 'Knowledge', y='V5',orient='h')
plt.xlabel('Knowledge', **axis_font)
plt.ylabel('Gender', **axis_font)

plt.subplot(1,2,2)
sns.boxplot(data=data, x= 'Practice', y='V5', orient='h')
plt.xlabel('Practice', **axis_font)
plt.ylabel('Gender', **axis_font)

plt.tight_layout()


# In[80]:


#Knowledge vs Gender
stat, p = mannwhitneyu(data.loc[data['V5'] == 'Male', 'Knowledge'], 
                       data.loc[data['V5'] == 'Female', 'Knowledge'], 
                       alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))

#Practice vs Gender
stat, p = mannwhitneyu(data.loc[data['V5'] == 'Male', 'Practice'], 
                       data.loc[data['V5'] == 'Female', 'Practice'],
                      alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))


# ***

# ### Type of farming vs Dependents

# In[81]:


# How many per group
data['V6'].value_counts()


# In[82]:


#Average practice and knowledge score for each group
data[['V6','Knowledge','Practice']].groupby('V6')['Knowledge', 'Practice'].agg(['count','mean','std'])


# In[83]:


axis_font = {'fontname':'Arial', 'size':'15'}
plt.subplots(1, 2, figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=data, x= 'Knowledge', y='V6',orient='h')
plt.xlabel('Knowledge', **axis_font)
plt.ylabel('Farming Type', **axis_font)

plt.subplot(1,2,2)
sns.boxplot(data=data, x= 'Practice', y='V6', orient='h')
plt.xlabel('Practice', **axis_font)
plt.ylabel('Farming Type', **axis_font)

plt.tight_layout()


# In[84]:


#Knowledge vs farming type
stat, p = mannwhitneyu(data.loc[data['V6'] == 'Commercial farming', 'Knowledge'], 
                       data.loc[data['V6'] == 'Subsistence farming', 'Knowledge'], 
                       alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))

#Practice vs farming type
stat, p = mannwhitneyu(data.loc[data['V6'] == 'Commercial farming', 'Practice'], 
                       data.loc[data['V6'] == 'Subsistence farming', 'Practice'],
                      alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))


# ***

# ### Previous training vs Dependents

# In[85]:


# How many have attended farmers training before
data['V7'].value_counts()


# In[86]:


#Average practice and knowledge score for each group
data[['V7','Knowledge','Practice']].groupby('V7')['Knowledge', 'Practice'].agg(['count','mean','std'])


# In[87]:


axis_font = {'fontname':'Arial', 'size':'15'}
plt.subplots(1, 2, figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=data, x= 'Knowledge', y='V7',orient='h')
plt.xlabel('Knowledge', **axis_font)
plt.ylabel('Previous Training', **axis_font)

plt.subplot(1,2,2)
sns.boxplot(data=data, x= 'Practice', y='V7', orient='h')
plt.xlabel('Practice', **axis_font)
plt.ylabel('Previous Training', **axis_font)

plt.tight_layout()


# In[88]:


#Knowledge vs previous training
stat, p = mannwhitneyu(data.loc[data['V7'] == 'Yes', 'Knowledge'], 
                       data.loc[data['V7'] == 'No', 'Knowledge'], 
                       alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))

#Practice vs previous training
stat, p = mannwhitneyu(data.loc[data['V7'] == 'Yes', 'Practice'], 
                       data.loc[data['V7'] == 'No', 'Practice'],
                      alternative='two-sided')
print('Statistics=%.3f \n p=%.3f' % (stat, p))


# ### Are those in farmers association more likely to have attended trainings in the past

# In[89]:


# Create crosstab of variables of interest
tab = pd.crosstab(data['V1'], data['V7'])


# In[90]:


tab


# In[91]:


from scipy.stats import chi2_contingency as chi2


# In[92]:


chi2(tab)


# #### Persons who belong to a farmers association are more likely to have attended a training before

# ### Are those visited by extension officers more likely to have attended trainings in the past

# In[93]:


tab2 = pd.crosstab(data['V3'], data['V7'])
chi2(tab2)


# #### Persons who have been visited by an extension officer were more likely to have attended a training before

# ***

# # Distribution of correct responses

# In[94]:


# How many persons got each knowledge question correct
knowledge_cols = ['K1','K2','K3','K4','K5','K6','K7','K8']
count=[]
percent=[]

for i in knowledge_cols:
    x= data[i].value_counts()[1]
    count.append(x)
    y= round((x/101)*100, 2)
    percent.append(y)


# In[95]:


knowledge = pd.DataFrame(list(zip(knowledge_cols, count, percent)),
              columns=['Question','N', '%'])


# In[96]:


knowledge


# In[97]:


# How many persons got each practice question correct
practice_cols = ['P1','P2','P3','P4','P5','P6','P7']
count2=[]
percent2=[]

for i in practice_cols:
    x1= data[i].value_counts()[1]
    count2.append(x1)
    y1= round((x1/101)*100, 2)
    percent2.append(y1)


# In[98]:


practice = pd.DataFrame(list(zip(practice_cols, count2, percent2)),
              columns=['Practice','N', '%'])


# In[99]:


practice


# # Relationship between Knowledge and Practice

# In[100]:


spearmanr(data['Knowledge'], data['Practice'])


# In[106]:


spearmanr(data['V30'], data['Practice'], nan_policy='omit')


# ---

# # Visualisation 

# In[101]:


plt.figure(figsize=(15,8))

plt.subplot(3, 3, 1)
sns.boxplot(data=data, x='Knowledge', y='V1')
plt.ylabel("")
plt.title('A: Belong to a farners associaton', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,2)
sns.boxplot(data=data, x='Knowledge', y='V3')
plt.ylabel("")
plt.title('B: Visited by an extension officer', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,3)
sns.boxplot(data=data, x='Knowledge', y='V4', 
            order=['below 20', '20-30','31-40','41-50','51-60','61 and above'])
plt.ylabel("")
plt.title('C: Age Group', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,4)
sns.boxplot(data=data, x='Knowledge', y='V5')
plt.ylabel("")
plt.title('D: Gender', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,6)
sns.distplot(data['Knowledge'], kde=False)
plt.title('E: Knowledge Score Distribution', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,7)
sns.boxplot(data=data, x='Knowledge', y='V6')
plt.ylabel("")
plt.yticks(rotation= 45)
plt.title('F: Farming Type', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,8)
sns.boxplot(data=data, x='Knowledge', y='V7')
plt.ylabel("")
plt.title('G: Previous Training', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,9)
sns.boxplot(data=data, x='Knowledge', y='Seen_One')
plt.ylabel("")
plt.title('H: Seen one symptom', fontsize=13, fontweight='bold', loc='left')


plt.tight_layout()

plt.savefig('Knowledge.png', dpi=500)


# In[102]:


plt.figure(figsize=(15,8))

plt.subplot(3, 3, 1)
sns.boxplot(data=data, x='Practice', y='V1')
plt.ylabel("")
plt.title('A: Belong to a farners associaton', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,2)
sns.boxplot(data=data, x='Practice', y='V3')
plt.ylabel("")
plt.title('B: Visited by an extension officer', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,3)
sns.boxplot(data=data, x='Practice', y='V4', 
            order=['below 20', '20-30','31-40','41-50','51-60','61 and above'])
plt.ylabel("")
plt.title('C: Age Group', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,4)
sns.boxplot(data=data, x='Practice', y='V5')
plt.ylabel("")
plt.title('D: Gender', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,6)
sns.distplot(data['Practice'], bins=5, kde=False)
plt.title('E: Practice Score Distribution', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,7)
sns.boxplot(data=data, x='Practice', y='V6')
plt.ylabel("")
plt.yticks(rotation= 45)
plt.title('F: Farming Type', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,8)
sns.boxplot(data=data, x='Practice', y='V7')
plt.ylabel("")
plt.title('G: Previous Training', fontsize=13, fontweight='bold', loc='left')

plt.subplot(3,3,9)
sns.boxplot(data=data, x='Practice', y='Seen_One')
plt.ylabel("")
plt.title('H: Seen one symptom', fontsize=13, fontweight='bold', loc='left')


plt.tight_layout()

plt.savefig('Practice.png', dpi=500)


# In[ ]:




