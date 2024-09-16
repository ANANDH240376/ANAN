import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data=pd.read_csv("yourdata.csv")
k = 2
# Create KMeans instance
kmeans = KMeans(n_clusters=k,random_state=0)
# Fit the model to the data
kmeans.fit(data)
cluster_assignments=kmeans.labels_
# Plot the data points and centroids
plt.scatter(data['Feature1'],data['Feature2'],c=cluster_assignments,cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
#show the graph
plt.show()
