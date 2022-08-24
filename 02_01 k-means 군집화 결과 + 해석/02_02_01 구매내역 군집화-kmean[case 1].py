# Data Analysis
import pandas as pd
import numpy as np

# Visualization
from sklearn.manifold import TSNE
from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# python modules
import os
import pickle
from tqdm import tqdm

# learning library
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

#%%
################# 기본 데이터. input ########################

os.chdir(r'C:\Users\9001283\Desktop\02-02 구매내역 군집화')

# data
data = pd.read_csv('02_01최종구매내역테이블.csv', index_col = 0)

# drop columns (분석제외.)
data = data.drop(['성별','나이','구매자_통신사','구매전환율'], axis = 1)


#%%
##### 구매-클릭비율 가지고만 k-mean #####

# buy와 click은 0~1사이로 인코딩 되어있다 = PCA시 표준화 필요 X
x = data.drop(['사용자ID','심야시간','주간시간','통근시간','구매가격'],axis = 1)
pca = PCA(n_components=10) 
pca_col = pca.fit_transform(x)


# 분산 설명 비율 - 1%까지는 살려왔다. 확인하면 적정 개수로는 5,10,15가 맞지 않나 싶음.
pca.explained_variance_ratio_

# 누적 분산 설명비율. = 10개로 축소한 이유.
pca.explained_variance_ratio_.sum()


pca_df = pd.DataFrame(data=pca_col, columns = ['pca_'+str(i) for i in range(1,11)])
# 주성분으로 이루어진 데이터 프레임 구성

# k-means 들어갈 데이터셋.
custom1_data = pd.concat([data.loc[:,['사용자ID','심야시간','주간시간','통근시간','구매가격']], pca_df],axis = 1)
custom1_data = custom1_data.set_index('사용자ID')

# 구매금액 표준화. 

# 표준화
scaler = StandardScaler() 
custom1_data.구매가격 = scaler.fit_transform(custom1_data.구매가격.to_frame())


del x, pca_col, pca_df

#%%
# 구매-클릭 비율 - 시간대 - k-means - 초안.
# 클러스터 1번 기준.


# 훈련과정 분할.
print('tSNE 차원축소 과정. 진행')
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, verbose = 1, n_jobs = -1,  random_state=23)
new_values = tsne_model.fit_transform(custom1_data)

df = pd.concat([pd.DataFrame(new_values),pd.Series(custom1_data.index)], axis = 1)
df.columns = ['x', 'y', 'book_name']

# 클러스터 개수.
range_n_clusters = [i for i in range(3,17)]


# 수정필요? 확인해볼것.
plt.figure(figsize=(10, 24)) 
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(custom1_data.to_numpy()) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(custom1_data.to_numpy())

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(custom1_data.to_numpy(), cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(custom1_data.to_numpy(), cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    df['clt_lab'] = cluster_labels
    
    ax2 = sns.scatterplot(
        x="x", y="y",
        hue='clt_lab',
        data=df,
        legend="full",
        alpha=0.5
        );
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    plt.legend(loc='lower right', labelspacing=0.15, ncol=2)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "(only rates PCA-10) Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

'''
실루엣 계수 결과저장.
n = 8이 제일 크게 등장함.

For n_clusters = 3 The average silhouette_score is : 0.21067935724348663
For n_clusters = 4 The average silhouette_score is : 0.25001246043312503
For n_clusters = 5 The average silhouette_score is : 0.2602991229148902
For n_clusters = 6 The average silhouette_score is : 0.26149384680219645
For n_clusters = 7 The average silhouette_score is : 0.2633504502007725
For n_clusters = 8 The average silhouette_score is : 0.2771013599459813
For n_clusters = 9 The average silhouette_score is : 0.258198288506233
For n_clusters = 10 The average silhouette_score is : 0.26596186053416204
For n_clusters = 11 The average silhouette_score is : 0.24067203705057078
For n_clusters = 12 The average silhouette_score is : 0.2689864877252248
For n_clusters = 13 The average silhouette_score is : 0.26738347746090974
For n_clusters = 14 The average silhouette_score is : 0.253846531693158
For n_clusters = 15 The average silhouette_score is : 0.25256452447231253
For n_clusters = 16 The average silhouette_score is : 0.25839824309465725
'''
plt.show()

#%%
######################## word2vec 추가 ######################################
# 추가로 word2vec 들어가면. 어떻게 되지? word2vec 으로 data 변경

# data load, w2v 결과 임포트.
with open('buy+click_word2vec_embedding.pkl', 'rb') as f:
    w2v_df = pickle.load(f)    

del f

data = pd.merge(data, w2v_df, how='left', left_on='사용자ID', right_on='user_id')
data = data.drop(['user_id'], axis = 1)

# k-means 들어갈 데이터셋.
custom2_data = data.set_index('사용자ID')

# 표준화
scaler = StandardScaler() 
custom2_data.구매가격 = scaler.fit_transform(custom2_data.구매가격.to_frame())

#%%

# 훈련과정 분할.
print('tSNE 차원축소 과정. 진행')
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, verbose = 1, n_jobs = -1,  random_state=23)
new_values = tsne_model.fit_transform(custom2_data)

df = pd.concat([pd.DataFrame(new_values),pd.Series(custom2_data.index)], axis = 1)
df.columns = ['x', 'y', 'book_name']

# 클러스터 개수.
range_n_clusters = [i for i in range(3,17)]

print('서브플롯 작업.')
# 수정필요? 확인해볼것.
plt.figure(figsize=(10, 24)) 
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(custom2_data.to_numpy()) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(custom2_data.to_numpy())

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(custom2_data.to_numpy(), cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(custom2_data.to_numpy(), cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    df['clt_lab'] = cluster_labels
    
    ax2 = sns.scatterplot(
        x="x", y="y",
        hue='clt_lab',
        data=df,
        legend="full",
        alpha=0.5
        );
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    plt.legend(loc='lower right', labelspacing=0.15, ncol=2)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "(rates + W2V full) Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


plt.show()
#%%


# buy와 click은 0~1사이로 인코딩 되어있다 = PCA시 표준화 필요 X
x = data.drop(['사용자ID','심야시간','주간시간','통근시간','구매가격'],axis = 1)
pca = PCA(n_components=60) 
pca_col = pca.fit_transform(x)

# 분산 설명 비율 - 1%까지는 살려왔다. 확인하면 적정 개수로는 5,10,15가 맞지 않나 싶음.
pca.explained_variance_ratio_

# 누적 분산 설명비율. = 10개로 축소한 이유.
pca.explained_variance_ratio_.sum()

pca_df = pd.DataFrame(data=pca_col, columns = ['pca_'+str(i) for i in range(1,61)])
# 주성분으로 이루어진 데이터 프레임 구성

# k-means 들어갈 데이터셋.
custom3_data = pd.concat([data.loc[:,['사용자ID','심야시간','주간시간','통근시간','구매가격']], pca_df],axis = 1)
custom3_data = custom3_data.set_index('사용자ID')


# 표준화
scaler = StandardScaler() 
custom3_data.구매가격 = scaler.fit_transform(custom3_data.구매가격.to_frame())




# 훈련과정 분할.
print('tSNE 차원축소 과정. 진행')
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, verbose = 1, n_jobs = -1,  random_state=23)
new_values = tsne_model.fit_transform(custom3_data)

df = pd.concat([pd.DataFrame(new_values),pd.Series(custom3_data.index)], axis = 1)
df.columns = ['x', 'y', 'book_name']

# 클러스터 개수.
range_n_clusters = [i for i in range(3,17)]

print('서브플롯 작업.')
# 수정필요? 확인해볼것.
plt.figure(figsize=(10, 24)) 
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(custom3_data.to_numpy()) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(custom3_data.to_numpy())

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(custom3_data.to_numpy(), cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(custom3_data.to_numpy(), cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    df['clt_lab'] = cluster_labels
    
    ax2 = sns.scatterplot(
        x="x", y="y",
        hue='clt_lab',
        data=df,
        legend="full",
        alpha=0.5
        );
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    plt.legend(loc='lower right', labelspacing=0.15, ncol=2)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "(rates + W2V PCA-60) Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()


#%%
# 어떤거 선택? 확인할지?