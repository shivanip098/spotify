import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix
from sklearn import datasets
from collections import Counter, defaultdict
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
from scipy.linalg import svd as scipy_svd
from mlxtend.frequent_patterns import apriori
import dataframe_image as dfi


#read in CSV
df = pd.read_csv('BB_Features.csv')
#for idx, column in enumerate(df.columns):
#    print(idx,column)
#focus_cols: 5:danceability, 6:energy, 7:key, 
#8:loudness, 9:mode, 10:acousticness, 11:instrumentalness,
#12: liveness, 13: valence, 14: tempo, 15: duration_ms, 
#16: time_signature




#Train/Test Split? 80:20?
train = df.sample(frac = 0.8)
 
# Creating dataframe with
# rest of the 50% values
test = df.drop(train.index)


#create updated df with relevant information/cols
df = df.iloc[:,5:16]
#print(df.head)

#Outlier detection - detect and exclude outliers
#https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe
df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
#print(df.shape)

#Scale/Normalize Dataset
df_scaled = df.copy()
scaled_col_dict = {}
# apply maximum absolute scaling
for column in df_scaled.columns:
    #keeping dict of scaled values in case I need to translate things back
    scaled_col_dict[column] = df_scaled[column].abs().max()
    df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    #print(column)

#Clustering dfs: key and mode, energy and valence, danceability and loudness
clus1 = df_scaled[['key', 'energy']].copy()
clus2 = df_scaled[['energy', 'valence']].copy()
clus3 = df_scaled[['danceability', 'loudness']].copy()

#K-means clustering --> 4 clusters
Kmean1 = KMeans(n_clusters=4)
Kmean1.fit(clus1)
print(Kmean1.cluster_centers_)

#K-means plot
clus1.plot.scatter(x='key', y='energy', title= "K Means: Key and Energy")
plt.scatter(0.08035991, 0.67415407, s=100, c='green', marker='s')
plt.scatter(0.88065489, 0.75941876, s=100, c='red', marker='s')
plt.scatter(0.49007845, 0.78661868, s=100, c='orange', marker='s')
plt.scatter(0.65656566, 0.46320585, s=100, c='yellow', marker='s')
plt.show()



clus1.plot.scatter(x='key', y='energy', title= "Scatter plot between variables Key and Energy")
plt.scatter(0.08035991, 0.67415407, s=100, c='green', marker='s')
plt.scatter(0.88065489, 0.75941876, s=100, c='red', marker='s')
plt.scatter(0.49007845, 0.78661868, s=100, c='orange', marker='s')
plt.scatter(0.65656566, 0.46320585, s=100, c='yellow', marker='s')
plt.savefig('Scatter_Key_Energy_KMeans', bbox_inches = 'tight')


#Normal and Truncated SVD - to reduce the dimensionality of data - n is sim of output data
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
#U is a m∗r orthonormal matrix of "left-singular" (eigen)vectors of AAT.
#V is a n∗r orthonormal matrix of 'right-singular' (eigen)vectors of ATA.
#Σ is a r∗r non-negative, decreasing order diagonal matrix. 
#All elements not on the main diagonal are 0 and the elements of Σ

u, s, v = scipy_svd(df_scaled)
#print("Eigenvectors of AAT: \n" u)
#print("Singular Values: \n", s)
#print("Eigenvectors of ATA: \n", v)
eigvals = s**2 / sum(s**2)

u_f, s_f, v_f = scipy_svd(df_scaled, full_matrices=False)
print("Eigenvectors of AAT: \n", u_f)
print("Singular Values: \n", s_f)
print("Eigenvectors of ATA: \n", v_f)
eigvals = s_f**2 / sum(s_f**2)

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(11) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (SVD)')
#I don't like the default legend so I typically make mine like below, e.g.
#with smaller fonts and a bit transparent so I do not cover up data, and make
#it moveable by the viewer in case upper-right is a bad place for it 
plt.savefig('SVD Scree Plot (Full Matrices = False)', bbox_inches = 'tight')


svd = TruncatedSVD(n_components=8, n_iter=10)
svd.fit(df_scaled)
print(svd.explained_variance_ratio_.sum())
s=svd.singular_values_
eigvals = s**2 / sum(s**2)

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(8) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (Truncated SVD)')
#I don't like the default legend so I typically make mine like below, e.g.
#with smaller fonts and a bit transparent so I do not cover up data, and make
#it moveable by the viewer in case upper-right is a bad place for it 
plt.savefig('Truncated SVD Scree Plot', bbox_inches = 'tight')

#Create Correlation Matrix and save
corrMtrx = df_scaled.corr()
#print(corrMtrx)
sns.heatmap(corrMtrx, annot=True, square=True)
#plt.figure(figsize=(12, 8))
plt.savefig('scaled_Correlation_Matrix', bbox_inches = 'tight')

#Create Pair Plot
#https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
sns.pairplot(df_scaled)
plt.savefig('scaled_Pair_Plot', bbox_inches = 'tight')





#Repeat all above steps but remove outliers and reduce dimensions:



#Ranking and Dimensionality Reduction- to rank importance and relevance of features
#PCA: - https://towardsdatascience.com/what-is-the-difference-between-pca-and-factor-analysis-5362ef6fa6f9
#https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
pca = PCA(0.95)
pca_fit = pca.fit(df_scaled)
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('PCA Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.savefig('scaled_PCA_Scree_Plot', bbox_inches = 'tight')

pca = PCA(0.95)
pca_fit = pca.fit(df_scaled)
print("The eigenvalues for the covariance matrix of our data are: ", list(np.round(pca_fit.explained_variance_, decimals=5)), end='\n\n')
pca_df = pd.DataFrame(abs(pca_fit.components_[:2]), columns = df_scaled.columns,
            index = ['Principal component 1', 'Principal component 2'])
print(pca_df.head)



X_std = StandardScaler().fit_transform(df_scaled)

pca = PCA(n_components=11)
pca_fit = pca.fit_transform(X_std)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


PCA_components = pd.DataFrame(pca_fit)

plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
#plt.savefig('scaled_PCA_Scatter_Plot', bbox_inches = 'tight')


#Truncated SVD - to reduce the dimensionality of data - n is sim of output data
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
svd_df = csr_matrix(df)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(svd_df)


#Factor Analysis - section unknown - figure out n components?
#https://towardsdatascience.com/what-is-the-difference-between-pca-and-factor-analysis-5362ef6fa6f9
#my_fa = FactorAnalysis(n_components=2)
# in new version of sklearn:
my_fa = FactorAnalysis(n_components=2, rotation='varimax') 
X_transformed = my_fa.fit_transform(df)

#Neural Networks - to discover hidden patterns and correlations → cluster/classify
#https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/



#elbow curve method:

#Compare K-means and Clustering for 3 different sets of variables:
#Create variable pair dfs
#K-means Clustering - to find groups within the data
#potential plots: violin, elbow, 


#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
#after inspecting elbow graph perform k-means:
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Scree: K-Means')
plt.show()

km = KMeans(n_clusters=2)
km.fit(df)

'''

#k means clustering --> results in array
km.cluster_centers_

#how to display cluster centroids (green/red) on plot
plt.scatter(df[ : , 0], df[ : , 1], s =50, c='b')
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()

'''

#https://www.kaggle.com/code/yugagrawal95/k-means-clustering-using-seaborn-visualization/notebook
#violin plot:
sns.violinplot('cylinders','mpg',data=df,palette='coolwarm')

# plots size by side where each plot is separated by col variable:
g = sns.FacetGrid(col='cylinders',data=df,legend_out=False)
g.map(sns.scatterplot,'hp','mpg')

#box plot of same data:
sns.boxplot('cylinders','time-to-60',data=df)

#new column that has cluster number for each row
df['cluster'] = kmeans.labels_


#Spectral Clustering - an alternative method of finding groups within the data
spec = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(df)

#Apriori - determine strength of association
df_copied = df_scaled.copy()
for x, column in enumerate(df_copied.columns):
    df_copied[str(column)] = np.where(df[str(column)] >= df[str(column)].mean(), True, False)

df_apri = apriori(df_copied, min_support=0.3, use_colnames=True)
df_apri['length'] = df_apri['itemsets'].apply(lambda x: len(x))
print(df_apri)




