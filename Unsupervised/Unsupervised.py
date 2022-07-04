# -*- coding: utf-8 -*-

# -- Sheet --

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as no
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# # 1 Data Loading and Cleaning


data = pd.read_csv("c_0000.csv")
data.head()
#Initial conditions. Col. 1, 2, 3: positions of stars; 4, 5, 6: velocities; 7: masses; 8: ids.

data.describe(include='all')

data.info()

no.bar(data, color='lightgreen')

for col in data.columns:
    print(data[col].value_counts())
    print("-" * 40)

# # 2 Data Visualisation


df1 = data.copy()
df1 = df1.drop(columns=['id',"m"])
for i in range(len(df1.columns)):
  plt.figure(figsize=(10,40)) # figure ration 16:9
  sns.set()
  plt.subplot(10, 1, i+1)
  sns.distplot(df1[df1.columns[i]], kde_kws={"color": "r", "lw": 3, "label": "KDE"}, hist_kws={"color": "b"})
  plt.title(df1.columns[i])

plt.figure(figsize=(20,15))
sns.heatmap(data[['x','y','z','vx','vy','vz','m']].corr(), annot = True) #overall correlation between the various columns present in our data
plt.title('Correlation Matrix', fontsize = 20)
plt.show()

# There is no correlation between those value, actually they each line represent a coordinate for a star and her velocity


# With our parameters we can explore some physics term like the time, distance


df = data.copy()
df['t'] = 0
df['v'] = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)
df['Ec'] = 0.5*data.m*(df.v)**2
df['r'] = np.sqrt(df.x**2 + df.y**2 + df.z**2)

for i in range(1,19):
    if i<10:
        step="0"+str(i)
    else:
        step=str(i)
    file="c_"+step+"00.csv"
    d=pd.read_csv(file)
    d['t']=i*100
    d['v'] = np.sqrt(d.vx**2 + d.vy**2 + d.vz**2)
    d['Ec'] = 0.5*d.m*(d.v)**2
    d['r'] = np.sqrt(d.x**2 + d.y**2 + d.z**2)
    df=df.append(d)

plt.figure(figsize=(20,20))
plt.scatter(df[df.t==0].x,df[df.t==0].y,s=1,marker='+')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cluster on the xy')
plt.grid()
plt.show()

plt.figure(figsize=(20,20))
plt.scatter(df.x,df.y,s=1,marker='+')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cluster on the xy at all time')
plt.grid()
plt.show()

# We can see that some stars are actually leaving our plan.We could actually perform In order to correctly use our cluster method we will use only our data at time 0. 


df_2 = pd.read_csv("c_1500.csv")

# # 3 Clustering


from sklearn.cluster import  KMeans
wcss = []

data2 = data.iloc[:,[0,1,2]]

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_) # inertia means that find to value of wcss

plt.figure(figsize=(10,10))
plt.plot(range(1,15),wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()

# It looks that 4 has an elbow point.


kmean2 = KMeans(n_clusters=4)
clusters = kmean2.fit_predict(data2)

data2["label"] = clusters
plt.figure(figsize=(10,10))
fig3 = plt.figure(figsize=(20,10))
ax = fig3.add_subplot(121, projection='3d')
ax.scatter(data2.x[data2.label == 0], data2.y[data2.label == 0],data2.z[data2.label== 0], color="red")
ax.scatter(data2.x[data2.label == 1], data2.y[data2.label == 1],data2.z[data2.label== 1],  color="blue")
ax.scatter(data2.x[data2.label == 2], data2.y[data2.label == 2],data2.z[data2.label== 2],  color="green")
ax.scatter(data2.x[data2.label == 3], data2.y[data2.label == 3],data2.z[data2.label== 3],  color="purple")

ax.scatter(kmean2.cluster_centers_[:,0],kmean2.cluster_centers_[:,1], color="orange") # scentroidler


# With the 3D reprensatation it's kinda hard to see everything so we use multiple angle


kmean2 = KMeans(n_clusters=4)
clusters = kmean2.fit_predict(data2)

data2["label"] = clusters
plt.figure(figsize=(10,10))
plt.scatter(data2.x[data2.label == 0], data2.y[data2.label == 0], color="red")
plt.scatter(data2.x[data2.label == 1], data2.y[data2.label == 1], color="blue")
plt.scatter(data2.x[data2.label == 2], data2.y[data2.label == 2], color="green")
plt.scatter(data2.x[data2.label == 3], data2.y[data2.label == 3], color="purple")

plt.scatter(kmean2.cluster_centers_[:,0],kmean2.cluster_centers_[:,1], color="orange") # scentroidler
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(data2.y[data2.label == 0], data2.z[data2.label == 0], color="red")
plt.scatter(data2.y[data2.label == 1], data2.z[data2.label == 1], color="blue")
plt.scatter(data2.y[data2.label == 2], data2.z[data2.label == 2], color="green")
plt.scatter(data2.y[data2.label == 3], data2.z[data2.label == 3], color="purple")

plt.scatter(kmean2.cluster_centers_[:,0],kmean2.cluster_centers_[:,1], color="orange") # scentroidler
plt.xlabel("y")
plt.ylabel("z")
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(data2.x[data2.label == 0], data2.z[data2.label == 0], color="red")
plt.scatter(data2.x[data2.label == 1], data2.z[data2.label == 1], color="blue")
plt.scatter(data2.x[data2.label == 2], data2.z[data2.label == 2], color="green")
plt.scatter(data2.x[data2.label == 3], data2.z[data2.label == 3], color="purple")

plt.scatter(kmean2.cluster_centers_[:,0],kmean2.cluster_centers_[:,1], color="orange") # scentroidler
plt.xlabel("x")
plt.ylabel("z")
plt.show()

# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans3 = KMeans(n_clusters=i)
    kmeans3.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.figure(figsize=(10,10))
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()

data3 = data2.iloc[:,data2.columns != 'label'].head(5000)

# dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(figsize=(10,10))
merg = linkage(data3, method="ward") # scipy is an algorithm of hiyerarchal clusturing
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

# HC
from sklearn.cluster import AgglomerativeClustering



hiyerartical_cluster = AgglomerativeClustering(n_clusters=4, affinity="euclidean",linkage="ward")
cluster = hiyerartical_cluster.fit_predict(data3)

data3["label"] = cluster
plt.figure(figsize=(10,10))
plt.scatter(data3.x[data3.label == 0], data3.y[data3.label == 0], color="red")
plt.scatter(data3.x[data3.label == 1], data3.y[data3.label == 1], color="blue")
plt.scatter(data3.x[data3.label == 2], data3.y[data3.label == 2], color="green")
plt.scatter(data3.x[data3.label == 3], data3.y[data3.label == 3], color="purple")

from sklearn.cluster import  KMeans
wcss = []

df_22 = df_2.iloc[:,[0,1]]

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_22)
    wcss.append(kmeans.inertia_) # inertia means that find to value of wcss

plt.figure(figsize=(10,10))
plt.plot(range(1,15),wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()

kmean2 = KMeans(n_clusters=4)
clusters = kmean2.fit_predict(df_22)

df_22["label"] = clusters
plt.figure(figsize=(15,15))
plt.scatter(df_22.x[df_22.label == 0], df_22.y[df_22.label == 0], color="red")
plt.scatter(df_22.x[df_22.label == 1], df_22.y[df_22.label == 1], color="blue")
plt.scatter(df_22.x[df_22.label == 2], df_22.y[df_22.label == 2], color="green")
plt.scatter(df_22.x[df_22.label == 3], df_22.y[df_22.label == 3], color="purple")

plt.scatter(kmean2.cluster_centers_[:,0],kmean2.cluster_centers_[:,1], color="orange") # scentroidler

plt.show()

# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans3 = KMeans(n_clusters=i)
    kmeans3.fit(df_22)
    inertia_list[i] = kmeans.inertia_
plt.figure(figsize=(10,10))
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()

df_3 = df_22.iloc[:,df_22.columns != 'label'].head(5000)

# dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(figsize=(10,10))
merg = linkage(df_3, method="ward") # scipy is an algorithm of hiyerarchal clusturing
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

hiyerartical_cluster = AgglomerativeClustering(n_clusters=4, affinity="euclidean",linkage="ward")
cluster = hiyerartical_cluster.fit_predict(df_3)

df_3["label"] = cluster
plt.figure(figsize=(10,10))
plt.scatter(df_3.x[df_3.label == 0], df_3.y[df_3.label == 0], color="red")
plt.scatter(df_3.x[df_3.label == 1], df_3.y[df_3.label == 1], color="blue")
plt.scatter(df_3.x[df_3.label == 2], df_3.y[df_3.label == 2], color="green")
plt.scatter(df_3.x[df_3.label == 3], df_3.y[df_3.label == 3], color="purple")



