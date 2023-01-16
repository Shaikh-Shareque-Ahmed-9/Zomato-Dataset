#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import seaborn as sns


# In[3]:


df=pd.read_csv("zomato.csv")
df


# # EDA

# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


df.info()


# # Restaurants delivering Online or not ?

# In[8]:


#Do they accept online order
counts = df['online_order'].value_counts()


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


fig = plt.gcf()
fig.set_size_inches(6, 6)
colors = ['#4F6272', '#B7C3F3']
_, _, autotexts = plt.pie(counts.values,labels = counts.index, autopct='%1.0f%%', pctdistance=0.8, 
                          labeldistance = 1.1,textprops={'fontsize': 16},explode = [0.15,0], colors = colors)

for ins in autotexts:
    ins.set_color('white')
plt.show()


# # Restaurants allowing table booking or no ?

# In[11]:


bookng = df['book_table'].value_counts()


# In[12]:


fig = plt.gcf()
fig.set_size_inches(6, 6)

_, _, autotexts = plt.pie(bookng.values,labels = bookng.index, autopct='%1.0f%%', pctdistance=0.8,
                          labeldistance = 1.1,textprops={'fontsize': 16},explode = [0.15,0])

for ins in autotexts:
    ins.set_color('white')
plt.show()


# # Table Booking Rating vs Other Ratings

# In[13]:


df['rate']


# In[14]:


#remove characters from rating and extract only rating integers
characters_to_remove = ['/5',' /5','NEW','-']
for i in characters_to_remove:
    df['rate'] = df['rate'].str.replace(i,'')
    
df['rate'].head()
#done replacing


# In[15]:


# replace NAN with mean value
df['rate'].fillna(0)
df['rate'] = pd.to_numeric(df["rate"], downcast="float")

df['rate'].replace(to_replace = 0, value = df['rate'].mean(), inplace=True)

#booking table rating

table_booking_rating_avg = df[df['book_table'] == 'Yes']['rate'].mean()
table_booking_rating_avg = round(table_booking_rating_avg,2)
print('Rating of Table Booking Orders is =',table_booking_rating_avg,'\n')
# non booking ratings

no_booking_rating_avg = df[df['book_table'] == 'No']['rate'].mean()
no_booking_rating_avg = round(no_booking_rating_avg,2)
print('Rating of No Table Booking Orders is =',no_booking_rating_avg)


# # Graphical Interpretation

# In[16]:


values = {'Ratings with Booking':[table_booking_rating_avg],'Ratings without Booking':[no_booking_rating_avg]}
data =pd.DataFrame.from_dict(values)
a4_dims = (6, 6)
fig , ax = plt.subplots(figsize=a4_dims)
sns.set(style='darkgrid')

sns.barplot(data = data,)
ax.set_xticklabels( ax.get_xticklabels(),rotation=45, fontsize = 16)
ax.set_ylabel(' Avg Ratings out of 5', fontsize = 16)
for i in ax.containers:
    ax.bar_label(i)


# # Find Best Location

# In[17]:


# For best location i think we should concider the top rating and sentiment analysis
#replace , and () round brackets in column names
df.columns = [
    c.replace(', ', '').replace('(', '_').replace(')', '') 
    for c in df.columns]

# top rating locations in each city

locations = df.groupby('location')['rate','votes','listed_in_city'].apply(lambda x: x.nlargest(10,columns = 'rate')).droplevel(1).reset_index()

locations.set_index('location')


# # Top 10 Cities

# In[18]:


# top 10 Cities
locations_city = df.groupby('location')['rate'].mean()
locations_city = pd.DataFrame(locations_city)
locations_city['loc_count'] = df['location'].value_counts()
top_10 = locations_city.sort_values(by=['loc_count', 'rate'],ascending=False).head(10)
top_10


# # Relation between Location and Rating

# In[19]:


# doesn't making sense for me the relation between a numeric value and string 
#value or may b i have a less knowledge yet, if someone guide plz comment


# # Restaurant Type

# In[20]:


# total 96 types so we show only top 10
types= df['rest_type'].value_counts().nlargest(10).reset_index()
types.index +=1
types


# In[21]:


a4_dims = (10, 6)
fig , ax = plt.subplots(figsize=a4_dims)
sns.set(style='darkgrid')
sns.barplot(data= types,y='index',x = 'rest_type',ax=ax)
ax.set_yticklabels( ax.get_yticklabels(), fontsize = 16)
ax.set_xlabel(' Restaurant Types', fontsize = 20)
ax.set_ylabel('')
for i in ax.containers:
    ax.bar_label(i)


# In[22]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))

df["location"].value_counts().plot(kind="bar") #top 10 locations based on counts

print('BTM is the best location based on number of restaurants since Other category is a mix of location with count less than 500')


#  **Relation between Location and Rating**

# In[23]:


plt.figure(figsize=(15,8))
sns.scatterplot(data=df, x='location', y='rate', hue = 'rate')
plt.xticks(rotation=90)
plt.grid()


# **Restaurant Type**

# In[24]:


df['rest_type'].value_counts()


# In[25]:


fig , ax = plt.subplots(figsize=(15,15))
df['rest_type'].value_counts().plot(kind="barh",color = sns.color_palette("dark"))
plt.title("Restaurant Types")
plt.grid(axis = 'x')


# **No. of restaurants in a Location**

# In[26]:


df['location'].value_counts().plot(kind='bar',figsize=(20,8),color = sns.color_palette("dark"))
plt.title('Number of Restaurants in a location - Top 10')
plt.grid(axis = 'y')


# **Most famous restaurant chains in Bengaluru**

# In[27]:


res_chain = df['rate'].groupby(df['name'],sort=True)
dict_2={}
for i,j in df['name'].value_counts()[:10].to_dict().items():
    dict_2[i]=round(res_chain.get_group(i).mean(),2)
cost_df2 = pd.DataFrame(list(dict_2.items()),columns=['Restaurant Name',"avg_rating"])


fig , ax = plt.subplots(figsize=(10,8))
sns.barplot(data = cost_df2.sort_values(by=['avg_rating'],ascending=False),
            x = 'avg_rating',y = 'Restaurant Name',palette=sns.blend_palette(colors, n_colors=10))
plt.title('Top 10 Restaurants by Average Ratings')
plt.grid(axis = 'x')


# **Top 10 by Value counts**

# In[28]:


plt.figure(figsize=(10,7))
df['name'].value_counts()[:10].plot(kind = 'bar', color = sns.color_palette("Paired"))
plt.title('Top 10 Restaurant Chains in Bangalore by number of outlets')
plt.xlabel('Restaurant name')
plt.ylabel("No of Outlets")
plt.grid(axis = 'y')
plt.xticks(rotation = 45)

