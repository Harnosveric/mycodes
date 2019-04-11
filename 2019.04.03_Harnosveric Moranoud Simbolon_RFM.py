
# coding: utf-8

# In[10]:


## LIBRARIES
import numpy as np
import pandas as pd
import time, warnings
warnings.filterwarnings("ignore")
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
get_ipython().magic('matplotlib inline')


## EXPLORING THE DATASET
retail_df = pd.read_excel("E:\Tes Data Science\Soal 2\Online Retail.xlsx", sheetname = "Online Retail")
retail_france = retail_df[retail_df["Country"]=="France"]
retail_france.shape
# remove the no-CustomerID
retail_france = retail_france.drop(retail_france[retail_france["CustomerID"].isnull() == True].index)
retail_france.shape

print("Summary..")
#exploring the unique values of each attribute
print("Number of transactions: ", retail_france["InvoiceNo"].nunique())
print("Number of products bought: ",retail_france["StockCode"].nunique())
print("Number of customers:", retail_france["CustomerID"].nunique() )
print("Percentage of customers NA: ", round(retail_france["CustomerID"].isnull().sum() * 100 / len(retail_df),2),"%" )


## RECENCY
# # To calculate recency, we need to choose a date point from which we evaluate how many days ago was the customer's last purchase.
# #last date available in our dataset
# retail_france['InvoiceDate'].max()
# now = dt.date(2011,12,9) # variabel now berdasarkan hasil sebelumnya yaitu 2011-12-9
# print(now)
# #create a new column called date which contains the date of invoice only
# retail_france['date'] = pd.DatetimeIndex(retail_france['InvoiceDate']).date
# #group by customers and check last date of purshace
# recency_df = retail_france.groupby(by='CustomerID', as_index=False)['date'].max()
# recency_df.columns = ['CustomerID','LastPurchaseDate']
# #calculate recency
# recency_df['Recency'] = recency_df['LastPurchaseDate'].apply(lambda x: (now - x).days)
# #drop LastPurchaseDate as we don't need it anymore
# recency_df.drop('LastPurchaseDate',axis=1,inplace=True)
# print(recency_df.head())

## FREQUENCY
# drop duplicates
retail_france_copy = retail_france
retail_france_copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
#calculate frequency of purchases
frequency_df = retail_france_copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
frequency_df.columns = ['CustomerID','Frequency']


## MONETARY
#create column total cost
retail_france['TotalCost'] = retail_france['Quantity'] * retail_france['UnitPrice']

monetary_df = retail_france.groupby(by='CustomerID',as_index=False).agg({'TotalCost': 'sum'})
monetary_df.columns = ['CustomerID','Monetary']
monetary_df.head()


## SEGMENTATION USE PARETO
# pareto_cutoff = 0.8*monetary_df["Monetary"].sum()
# if monetary_df["Monetary"].cumsum() <= pareto_cutoff:
#     print("Top 20 %")
# else:
#     print("Bottom 80%")
    
    
## CREATE RFM TABLE
# #merge recency dataframe with frequency dataframe
# temp_df = recency_df.merge(frequency_df,on='CustomerID')
# temp_df.head()
# merge with monetary dataframe to get a table with the 3 columns
rfm_df = frequency_df.merge(monetary_df,on='CustomerID')
#use CustomerID as index
rfm_df.set_index('CustomerID',inplace=True)
#check the head
print(rfm_df.head())

# save RFM Table to CSV format
rfm_df.to_csv(r'E:\Tes Data Science\Soal 2\2019.04.03_Harnosveric Moranoud Simbolon_RFM.csv')


# In[6]:


monetary_df["Monetary"].sum()

