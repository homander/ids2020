import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#############
# Load data #
#############

df = pd.read_csv('processed-data/df_filtered.csv', index_col=0)
X = pd.read_csv('processed-data/2020-X.csv', index_col=0)

###################
# With 3 clusters #
###################

# 30 seems sufficient, but let's use 100 to be safe
N_INIT = 100

kmeans = KMeans(init='k-means++', n_clusters=3, n_init=N_INIT)
kmeans.fit(X)

res = pd.DataFrame()
res['Job Type'] = df['Job Type']
res['Label'] = kmeans.labels_

print(res)

scientist = res[res['Job Type'] == 'data scientist']
engineer = res[res['Job Type'] == 'data engineer']
analyst = res[res['Job Type'] == 'data analyst']

f, ax = plt.subplots(1, 3, sharey=False)
ax[0].hist(scientist['Label'])
ax[0].set_title('Data Scientist Jobs')
ax[0].set_xlabel('Label')
ax[0].set_ylabel('Number of jobs')
ax[1].hist(engineer['Label'])
ax[1].set_title('Data Engineer Jobs')
ax[1].set_xlabel('Label')
#ax[1].set_ylabel('Number of jobs')
ax[2].hist(analyst['Label'])
ax[2].set_title('Data Analyst Jobs')
ax[2].set_xlabel('Label')
#ax[2].set_ylabel('Number of jobs')
plt.show()
