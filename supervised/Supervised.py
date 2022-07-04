# -*- coding: utf-8 -*-

# -- Sheet --

#Import library
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import missingno as no
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# # 1. Data Loading and Cleaning


df_data = pd.read_csv("twitchdata-update.csv")
df_data.insert(0, 'id', range(1, 1 + len(df_data))) #Create ID colulmn
df_data

df_data.info()

df_data.describe(include='all')

no.bar(df_data, color='lightgreen')

# There is no NULL value in this dataframe so we can use every line


# ## 2. Data Visualisation


# We started this part by drawing some histogrammes to get a better view at our data


plt.figure(figsize=(20,10))
df_data['Stream time(minutes)'].head(20).plot.bar(color='orangered')
plt.xlabel('Streamers n°')
plt.ylabel('Time(in minutes)')
plt.title("Temps de Stream pour le Top 20")
plt.show()

df1 = df_data.copy()
df1 = df1.drop(columns=['Channel','id',"Language"])
for i in range(len(df1.columns)):
  plt.figure(figsize=(15,40)) # figure ration 16:9
  sns.set()
  plt.subplot(10, 1, i+1)
  sns.distplot(df_data[df1.columns[i]], kde_kws={"color": "r", "lw": 3, "label": "KDE"}, hist_kws={"color": "b"})
  plt.title(df1.columns[i])

# Next, we wanted to take a look at the language column since we did not represented it


plt.figure(figsize=(20,10))
plot = sns.countplot(x="Language", data=df_data,order=df_data['Language'].value_counts().index, palette="Set2")

# In the end, the histogram representation was not good for us. We had some issue since we wanted a clear representation. We opted for a pie graph to have a better representation


df = df_data.copy()
df.loc[(df["Language"] != "English") & (df["Language"] != "French")& (df["Language"] != "Korean")& (df["Language"] != "Russian")& (df["Language"] != "Spanish"), "Language"] = "Else"
plt.figure(figsize=(20,16))
df1 = df['Language'].value_counts()
plt.pie(df1.values, labels=df1.index,autopct='%0.1f%%')
plt.title('Percentage of Language', fontsize=15)
plt.show()

# To finish this part we made 2 wordcloud for thee channel and language representation and establish the correlation matrix


from wordcloud import WordCloud
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(
                          background_color='Black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df_data.Channel))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

from wordcloud import WordCloud
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(
                          background_color='Black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df_data.Language))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

plt.figure(figsize=(20,15))
sns.heatmap(df_data[['Channel', 'Watch time(Minutes)', 'Stream time(minutes)', 'Followers','Peak viewers','Average viewers','Followers gained','Views gained','Partnered','Mature','Language']].corr(), annot = True) #overall correlation between the various columns present in our data
plt.title('Correlation Matrix', fontsize = 20)
plt.show()

# For the correlation matrix we can clearly see some correlation with scores likes 0,72 or 0,68


# # 3. Regression Algorithms


def test_model(model):
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

# We start by split our data with train_test_split from sklearn


y = df_data['Followers gained']
X = pd.read_csv("twitchdata-update.csv")
X = X.drop(columns=['Channel', 'Language', 'Followers gained', 'Partnered', 'Mature'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=46)
X

# ## 3.1 Linear Regression


lr = LinearRegression()
lr.fit(X_train, y_train)
ypred_train = lr.predict(X_train)
ypred_test = lr.predict(X_test)
scores = cross_val_score(lr, X, y, cv=10)
print("Cross Validation: ",scores)
ac1 = metrics.r2_score(y_test, ypred_test)*100
print("Accuracy of training data", metrics.r2_score(y_train, ypred_train)*100)
print("Accuracy of testing data:", ac1)

# ## 3.2 RandomForest


Rfr = RandomForestRegressor(n_estimators = 100,max_depth=3, max_features="auto", min_samples_split = 3)
Rfr.fit(X_train, y_train)
ypred_train = Rfr.predict(X_train)
ypred_test = Rfr.predict(X_test)
ac2 = metrics.r2_score(y_test, ypred_test)*100
scores_regr = metrics.mean_squared_error(y_test, ypred_test)
scores = cross_val_score(Rfr, X, y, cv=10)
print("Cross Validation: ",scores)
print("Cross Validation: ",scores_regr)
print("R2 score of training data", metrics.r2_score(y_train, ypred_train)*100)
print("R2 score of testing data:", ac2)

# ## 3.3 Gradient Boosting Regressor


Gbr = GradientBoostingRegressor(learning_rate=0.05, n_estimators = 90, subsample = 0.9)
Gbr.fit(X_train, y_train)
ypred_train = Gbr.predict(X_train)
ypred_test = Gbr.predict(X_test)
scores = cross_val_score(Gbr, X, y, cv=10)
print("Cross Validation: ",scores)
ac3 = metrics.r2_score(y_test, ypred_test)*100
print("Accuracy of training data", metrics.r2_score(y_train, ypred_train)*100)
print("Accuracy of testing data:", ac3)

# ## 3.4 Final Result


accuracy =  {ac1: 'Logistic Regression', ac2: 'Random Forest', ac3:'Gradient Boosting'}
sns.set_style('darkgrid')
plt.figure(figsize=(14, 10))
model_accuracies = list(accuracy.values())
model_names = list(accuracy.keys())
sns.barplot(x=model_accuracies, y=model_names, palette='rainbow')

# In term of accuracy we have GBR (gradient bossting regressor) that is better thn the Logistic Regression and the Random Forest


