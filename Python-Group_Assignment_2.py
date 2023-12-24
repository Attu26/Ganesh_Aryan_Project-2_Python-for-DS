#!/usr/bin/env python
# coding: utf-8

# 
# Import the required libraries and load the data

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# In[2]:


df = pd.read_csv('download-11.csv')
df.head()


# 
# Column Unnamed Looks unnecessary, so will drop it beofore we go forward.

# In[3]:


df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()


# # Dataset Description:
# 
# The dataset contains measurements of clothing fit from RentTheRunway. RentTheRunWay is a unique platform that allows women to rent clothes for various occasions. The collected data is of several categories. This dataset contains self-reported fit feedback from customers as well as other side information like reviews, ratings, product categories, catalog sizes, customers’ measurements (etc.)

# # Attribute Information:
# 
# user_id: a unique id for the customer
# 
# item_id: unique product id
# 
# weight:weight measurement of customer
# 
# rented for:purpose clothing was rented for
# 
# body type:body type of customer
# 
# review_text:review given by the customer
# 
# size:the standardized size of the product
# 
# rating:rating for the product
# 
# age:age of the customer
# 
# category:the category of the product
# 
# bust size:bust measurement of customer
# 
# height:height of the customer
# 
# review_date:date when the review was written
# 
# fit:fit feedback

# # Project Objective:
# 
# Based on the given users and items data of an e-commerce company, segment the similar user and items into suitable clusters. Analyze the clusters and provide your insights to help the organization promote their business.

# Checking the first few samples, shape, info of the data to familiarize with different features.

# In[4]:


df.shape


# In[5]:


df.info()

*There are 192544 rows and 15 columns.
*We can observe the missing values in the dataset
*There are around 10 object type variables and 5 numerical variables
# In[6]:


## checking the presence of duplicate records
len(df[df.duplicated()])


# There are 189 duplicated values in it, will go ahead and drop them

# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


#Will check dupe values just to be sure
len(df[df.duplicated()])


# As seen above, there are no duplicated valus now.
# Drop the columns which you think redundant for the analysis.(Hint: drop columns like ‘id’, ‘review’)
# In[9]:


## will go ahead and remove redundant columns from the dataset.
#the below columns seems to okay to drop as it wont impact much on the modle.
df.drop(['user_id', 'item_id', 'review_text', 'review_summary', 'review_date'],axis=1,inplace=True)


# Checking first rows after droping the above columns

# In[10]:


df.head()


# Check the column 'weight', Is there any presence of string data? If yes, remove the string data and convert to float. (Hint: 'weight' has the suffix as lbs)

# In[11]:


df['weight'] = df['weight'].str.replace('lbs','').astype(float)


# In[12]:


df['weight'].head()


# Check the unique categories for the column 'rented for' and group 'party: cocktail' category with 'party'.

# In[13]:


#checking unique category for column 'rented for'
df['rented for'].unique()


# In[15]:


## grouping 'party: cocktail' category with the 'party'.
df['rented for'] = df['rented for'].str.replace('party: cocktail','party')


# In[16]:


## recheck unique values after grouping
df['rented for'].unique()

The column 'height' is in feet with a quotation mark, Convert to inches with float datatype.
# In[17]:


## Removing quotation marks
df['height'] = df['height'].str.replace("'",'')
df['height'] = df['height'].str.replace('"','')


# In[18]:


## Convert the feet to inches and convert the datatype to float
df['height'] = (df['height'].str[:1].astype(float)*12 + df['height'].str[1:].astype(float))


# In[19]:


df['height'].head()


# ###### Check for missing values in each column of the dataset? If it exists, impute them with appropriate methods.

# In[20]:


df.isnull().sum()/len(df)*100


# ##### Missing values are presented in most of columns, except size, fit and category.

# In[21]:


## As columns in question are in numerical so will use median strategy to impute.
for col in ['weight','rating','height','age']:
    df[col].fillna(df[col].median(), inplace=True)


# In[22]:


## Lets treat categoricak columns with mode imputation technique.
for col in ['bust size','rented for','body type','category']:
    df[col].fillna(df[col].mode()[0], inplace=True)


# In[23]:


df.isnull().sum()/len(df)*100


# All Null values are now imputed with Median values.

# #### Check the statistical summary for the numerical and categorical columns and write your findings.

# In[24]:


##checking the statistical summary for the numerical columns
df.describe().T


# In[25]:


## checking the statistical summary for the categorical columns.
df.describe(include='O').T


# -The average weight of the customer is around 137lbs.
# 
# -The average rating is around 9.
# 
# -The maximum height of the customer is 78 inches.
# 
# -The maximum standarized size of the product is 58.
# 
# -The age range is 0 to 117.
# 
# -Note we can see the min age is 0 we need to impute it with appropriate value and the maximun age we need to cap it to Upperlimit.
# 
# -Most of the customers rented the product for wedding and the most appeared product category is as dress.

# #### Are there outliers present in the column age? If yes, treat them with the appropriate method.

# In[26]:


sns.boxplot(df['age'])
plt.show()


# As we can see, outliers are present in 'Age" column.
# 
# Will treat them with capping technique to avoid loss of data.

# In[27]:


## lets treat the outliers in the column age using capping techinque

df['age'] = pd.DataFrame(np.where(df['age']>=100,100,df['age']))
df['age'] = pd.DataFrame(np.where(df['age']<=20,20,df['age']))


# In[28]:


sns.boxplot(df['age'])
plt.show()


# In[29]:


## after applying capping technique for the column age, there might be some presence of missing values in columns age, So drop them
df.dropna(inplace=True)


# ##### Check the distribution of the different categories in the column 'rented for' using appropriate plot.

# In[30]:


#checking for the distribution of the column rented for
sns.countplot(df['rented for'])
plt.xticks(rotation=45)
plt.show()


# Most of the customers have rented the product for the 'wedding' followed by'formal affair'& 'party'.

# ##### Data Preparation for model building

# #### Encode the categorical variables in the dataset.

# In[31]:


## will make a copy of the cleaned dataset before encoding and standardizing the columns
dfc1 = df.copy()


# In[32]:


## Encoding categorical variables using label encoder

## select object datatype variables
object_type_variables = [i for i in df.columns if df.dtypes[i] == object]
object_type_variables 


le = LabelEncoder()

def encoder(df):
    for i in object_type_variables:
        q = le.fit_transform(df[i].astype(str))  
        df[i] = q                               
        df[i] = df[i].astype(int)
encoder(df)


# In[33]:


df.head()


# Standardize the data, so that the values are within a particular range.

# In[34]:


## Tranforming the data using minmax scaling approach so that the values range will be 1.

mm = MinMaxScaler()

df.iloc[:,:] = mm.fit_transform(df.iloc[:,:])
df.head()


# ###### Principal Component Analysis and Clustering:
# 
# 

# ###### Apply PCA on the above dataset and determine the number of PCA components to be used so that 90-95% of the variance in data is explained by the same

# In[36]:


## will make another copy of the transformed dataset after encoding and standardizing the columns.
dfc2 = df.copy()


# In[37]:


## step1: Calculate the covariance matrix.
cov_matrix = np.cov(df.T)
cov_matrix


# In[38]:


## step2: Calculate the eigen values and eigen vectors.
eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
print('eigein vals:','\n',eig_vals)
print('\n')
print('eigein vectors','\n',eig_vectors)


# In[40]:


## step3: Scree plot.
total = sum(eig_vals)
var_exp = [(i/total)*100 for i in sorted(eig_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('Explained Variance: ',var_exp)
print('Cummulative Variance Explained: ',cum_var_exp)


# In[41]:


## Scree plot.
plt.bar(range(10),var_exp,align='center',color='lightgreen',edgecolor='black',label='Explained Variance')
plt.step(range(10),cum_var_exp,where='mid',color='red',label='Cummulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explianed Variance ratio')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.show()


# As noticed from the above scree plot the first 6 principal components are explaining the about 90-95% of the variation, So we can choose optimal number of principal components as 6.

# ###### Apply K-means clustering and segment the data. (You may use original data or PCA transformed data)
# 
# a. Find the optimal K Value using elbow plot for K Means clustering.
# 
# b. Build a Kmeans clustering model using the obtained optimal K value from the elbow plot.
# 
# c. Compute silhouette score for evaluating the quality of the K Means clustering technique.

# In[42]:


## Using the dimensions obtainted from the PCA to apply clustering.(i.e, 6)
pca = PCA(n_components=6)

pca_df = pd.DataFrame(pca.fit_transform(df),columns=['PC1','PC2','PC3','PC4','PC5','PC6'])
pca_df.head()


# New dimensions obtained from the application of PCA.

# ###### Kmeans clustering (using the PCA tranformed data)

# In[43]:


## finding optimal K value by KMeans clustering using Elbow plot.
cluster_errors = []
cluster_range = range(2,15)
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters,random_state=100)
    clusters.fit(pca_df)
    cluster_errors.append(clusters.inertia_)


# In[45]:


## creataing a dataframe of number of clusters and cluster errors.
cluster_df = pd.DataFrame({'num_clusters':cluster_range,'cluster_errors':cluster_errors})


# In[46]:


## Elbow plot.
plt.figure(figsize=[15,5])
plt.plot(cluster_df['num_clusters'],cluster_df['cluster_errors'],marker='o',color='b')
plt.show()


# From the above elbow plot we can see at the cluster K=3, the inertia significantly decreases . Hence we can select our optimal clusters as K=3.

# In[47]:


## Applying KMeans clustering for the optimal number of clusters obtained above.
kmeans = KMeans(n_clusters=3, random_state=100)
kmeans.fit(pca_df)


# In[49]:


## creating a dataframe of the labels.
label = pd.DataFrame(kmeans.labels_,columns=['Label'])


# In[50]:


## joining the label dataframe to the pca_df dataframe.
kmeans_df = pca_df.join(label)
kmeans_df.head()


# In[52]:


kmeans_df['Label'].value_counts()


# In[53]:


## finding optimal clusters through silhoutte score
from sklearn.metrics import silhouette_score
for i in range(2,15):
    kmeans = KMeans(i,random_state=100)
    kmeans.fit(pca_df)
    labels = kmeans.predict(pca_df)
    print(i,silhouette_score(pca_df,labels))


# -Above from elbow plot we chose optimal K value as 3 and we built a Kmeans clustering model.
# 
# -From the silhoutte score we can observe the for clusters 2 and 3 the score is higher. We can build Kmeans clustering model using the optimal K value as either 2 or 3.

# ###### Apply Agglomerative clustering and segment the data. (You may use original data or PCA transformed data)
# 
# a. Find the optimal K Value using dendrogram for Agglomerative clustering.
# 
# b. Build a Agglomerative clustering model using the obtained optimal K value observed from dendrogram.
# 
# c. Compute silhouette score for evaluating the quality of the Agglomerative clustering technique. (Hint: Take a sample of the dataset for agglomerative clustering to reduce the computational time)

# Agglomerative clustering (using original data)

# #Let us use the dfc2 for this (a copy of the cleaned dataset after encoding and data standardization)
# 
# #Since dataset is huge plotting dendrogram might be time consuming.
# 
# #Let us take a sample of the dataset. (since the dataset is huge around 2 lakh rows, let take a sample)

# In[54]:


## Taking a sample of 50K rows from the dfc2 dataframe using random sampling technique provided by pandas 

## Storing it in the new dataframe called 'dfc3' 
dfc3 = dfc2.sample(n=50000)

## reseting the index
dfc3.reset_index(inplace=True,drop=True)


# In[55]:


dfc3.head(4)


# In[ ]:


plt.figure(figsize=[18,5])
merg = linkage(dfc3, method='ward')
dendrogram(merg, leaf_rotation=90,)
plt.xlabel('Datapoints')
plt.ylabel('Euclidean distance')
plt.show()


# We look for the largest distance that we can vertically observe without crossing any horizontal line.
# 
# We can observe from the above dendrogram that we can choose optimal clusters has 2.

# In[ ]:


## Building hierarchical clustering model using the optimal clusters as 2
hie_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                                     linkage='ward')
hie_cluster_model = hie_cluster.fit(dfc3)


# In[ ]:


## Creating a dataframe of the labels
df_label1 = pd.DataFrame(hie_cluster_model.labels_,columns=['Labels'])
df_label1.head(5)


# In[ ]:


## joining the label dataframe with unscaled initial cleaned dataframe.(dfc1)

df_hier = dfc1.join(df_label1)
df_hier.head()


# In[ ]:


for i in range(2,15):
    hier = AgglomerativeClustering(n_clusters=i)
    hier = hier.fit(dfc3)
    labels = hier.fit_predict(dfc3)
    print(i,silhouette_score(dfc3,labels))


# *We can observe from the silhouette scores for the agglomerative clustering for the 2 clusers the silhouette score is higher.

# # 17. Conclusion.

# Perform cluster analysis by doing bivariate analysis between cluster label and different features and write your conclusion on the results.

# In[ ]:


df_hier.head(2)


# In[ ]:


df_hier['Labels'].value_counts().plot(kind='pie',autopct='%0.1f')
plt.show()


# We can observe that the clusters formed are imbalanced. There are more number of records assigned to cluster 0 than that of cluster 1.

# In[ ]:


## Let us check the distribution of the different categories of 'rented for' column
## w.r.t the clusters formed by agglomerative clustering technique.
sns.countplot(df_hier['rented for'],hue='Labels',data=df_hier)
plt.xticks(rotation = 45)
plt.show()


# We can observe that there are more number of users who have rented the product is for 'wedding' and also there are more number of users belong to the cluster 0 compare to the cluster 1.

# In[ ]:


## Lets check the age distribution of the different clusters.
sns.kdeplot(df_hier['age'],hue='Labels',data=df_hier)
plt.show()


# *The distribution of the varibale 'age' for different clusters is almost same, as there are more number of observations assigned to the cluster 0.
# 
# *We tried to implement and apply PCA on the renttherunway dataset and we selected 6 PCA compoments, which given us the 90-95% of the variance in the data.
# 
# *We used the PCA dimensions to cluster the data and segment the similar data in to clusters using KMeans clustering.
# 
# *We did Kmeans clustering algorithm to cluster the data, First we chose the optimal K value with the help of elbow plot used obtained K value from elbow plot to build a kmeans clustering model.
# 
# *We applied silhoutte score for the different K values and evaluated the goodness of the clustering technique used.
# 
# *We took the sample of the data and did agglomerative clustering using the original data and plotted dendrogram and analyzed the optimal number of classes and built a agglomerative clustering model using the obtained K value and evaluated the model using silhoutte score.
# 
# *In this dataset, we had less number of features, further we can ask the company to collect the demographic information such as income and education. Geographic info such as where the customer is located rural or urban, city etc. Behavioral info such as browsing, spent amount by category, sentiment towards specific products and price points, and lastly the survey on lifestyle info such as hobbies, fashion etc.
# 
# *By collecting more features, the customer segmentation/clustering of similar customers into groups will be more effective and we can infer more out of the clusters formed and will able to give suggestions to the company based on the analysis that will help the business to target the right customers and stand in the market for longer and make high revenue.

# In[ ]:




