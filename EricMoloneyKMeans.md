This post is a follow on of my previous paper where I introduced Python and different methods to download stock data. With the downloaded data we are going to calculate three features, historic returns, volatility and correlation of the stocks in the portfolio to the SPY, an ETF of the S&P500 stock index, which is a measure of the US stock market. I will then proceed to use the K-Means clustering algorithm to divide the stocks into distinct groups based upon said features. Dividing assets into groups with similiar characteristics can help construct diversified, long/short or mean reverting portfolios to name a few.


![image](https://user-images.githubusercontent.com/35773761/39299112-e61fb0d6-493f-11e8-8f02-76139b2d401e.png)

K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. Rather than defining groups before looking at the data, clustering allows you to find and analyze the groups that have formed organically. Each centroid of a cluster is a collection of feature values which define the resulting groups. Examining the centroid feature weights can be used to qualitatively interpret what kind of group each cluster represents. The Κ-means clustering algorithm uses iterative refinement to produce a final result. The algorithm inputs are the number of clusters Κ and the data set.

In the code that follows I run through necesssary steps for data collection, manipulationand analysis. First things first, we need to import python packages and the stock data from a csv file.

```
# import necessary packages
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import datetime as dt
from datetime import datetime

#reading in saved stock data from csv file
df = pd.read_csv("ETF Test Clean.csv")
df['date']=pd.to_datetime(df['date'])
df
```


![image](https://user-images.githubusercontent.com/35773761/39311187-23348bf8-4964-11e8-8b9e-a65bab496df0.png)

We can now analyse the data for our K-Means investigation. We need to decide how many clusters we want for the data. To do this we plot an “Elbow Curve” to highlight the relationship between how many clusters we choose, and the Sum of Squared Errors (SSE) resulting from using that number of clusters. From this plot we identify the optimal number of clusters to use – we would prefer a lower number of clusters, but also would prefer the SSE to be lower – so this trade off needs to be taken into account. In my analysis I look the point at the center to the elbow.
 ```
 #Calculate average percentage return and volatilities from 2017-01-03 to 2018-03-02
ret = df.pct_change()
returns = df.pct_change().mean() * len(df)
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = df.pct_change().std() * sqrt(len(df))
returns["CorrSPY"]=ret.corr()["SPY"]
#format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility']),np.asarray(returns['CorrSPY'])]).T

X = data
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
 
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()
```

![image](https://user-images.githubusercontent.com/35773761/39311460-e33ef91a-4964-11e8-9ff1-01ffe12fccb4.png)





