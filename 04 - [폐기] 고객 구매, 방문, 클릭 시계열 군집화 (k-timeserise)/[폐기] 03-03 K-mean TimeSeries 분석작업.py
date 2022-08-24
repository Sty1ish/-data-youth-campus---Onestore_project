###############################
#####  경로, 파일, 패키지  #####
###############################
import os 
from tqdm import tqdm
import math
import datetime

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# K-mean clustering packages
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# default setting + Data import

os.chdir(r'C:\Users\9001283\Desktop\download_pts.onestorecorp.com(2022-08-02)')
click_data = pd.read_csv('산학_클릭로그.csv', header = None, names = ['거래일시','거래시간','유저ID','대표상품ID'])

# sort table
click_data = click_data.sort_values(by=['거래일시', '거래시간'] ,ascending=True).reset_index(drop=True)

# cross_table (row = time [==index col] , col = user)
click_data = pd.crosstab(click_data['거래일시'], click_data['유저ID'])


# 데이터 프레임 재설정.
mySeries = []
namesofMySeries = []

for i in click_data.columns:
    df = click_data[i].to_frame()
    mySeries.append(df)
    namesofMySeries.append(str(i))
   
    
   
# 상위 48개 정도 봐보자, rp cp, fig size 변경으로 그려볼수 있다.
rp, cp = 6, 8
fig, axs = plt.subplots(rp,cp,figsize=(30,11))
fig.suptitle('Series')
for i in range(rp):
    for j in range(cp):
        if i*cp+j+1>len(mySeries): # pass the others that we can't fill
            continue
        axs[i, j].plot(mySeries[i*cp+j].values)
        axs[i, j].set_title(namesofMySeries[i*cp+j])
plt.show()


# 각 시리즈의 길이는 31이다. 한달중 결측이 없기에 문제없는 데이터셋.
# 따로 전처리를 필요로 하지는 않는다.
series_lengths = {len(series) for series in mySeries}
print(series_lengths)


# K-mean 데이터셋은 스케일링에 크게 영향을 받는다. 따라서 모든 데이터 셋에 min-max scaleing을 실시해준다.
for i in range(len(mySeries)):
    scaler = MinMaxScaler()
    mySeries[i] = MinMaxScaler().fit_transform(mySeries[i])
    mySeries[i]= mySeries[i].reshape(len(mySeries[i]))


# 결과 형태 출력
print("max: "+str(max(mySeries[0]))+"\tmin: "+str(min(mySeries[0])))
print(mySeries[0][:5])



# 클러스터링 방법 1 SOM 방법

som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
# I didn't see its significance but to make the map square,
# I calculated square root of map size which is 
# the square root of the number of series
# for the row and column counts of som

# sigma: 0.3
# learning_rate: 0.5
# random weight initialization
# 50.000 iteration
# Map size: square root of the number of series
# As a side note, I didn't optimize these parameters due to the simplicity of the dataset.
som = MiniSom(som_x, som_y,len(mySeries[0]), sigma=0.3, learning_rate = 0.1)

som.random_weights_init(mySeries)
som.train(mySeries, 50000)


# Little handy function to plot series
def plot_som_series_averaged_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
    fig.suptitle('Clusters')
    for x in tqdm(range(som_x)):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()
    


win_map = som.win_map(mySeries)
# Returns the mapping of the winner nodes and inputs

plot_som_series_averaged_center(som_x, som_y, win_map)



def plot_som_series_dba_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
    fig.suptitle('Clusters')
    for x in tqdm(range(som_x)):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") # I changed this part
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()



win_map = som.win_map(mySeries)

plot_som_series_dba_center(som_x, som_y, win_map)


# 뭐한거임 ㅋㅋㅋㅋㅋ...
# 2. 3. 1. 2. Cluster Distribution¶
# We can see the distribution of the time series in clusters in the following chart.



# Let's check first 5
for series in mySeries[:5]:
    print(som.winner(series))



cluster_map = []
for idx in range(len(mySeries)):
    winner_node = som.winner(mySeries[idx])
    cluster_map.append((namesofMySeries[idx],f"Cluster {winner_node[0]*som_y+winner_node[1]+1}"))

pd.DataFrame(cluster_map,columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")






# k-mean 클러스터링

cluster_count = math.ceil(math.sqrt(len(mySeries))) 
# A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN


# 임의로 36개의 그룹으로 나눠버림, 
cluster_count = 36
km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw", verbose = 1, n_jobs = -1)

labels = km.fit_predict(mySeries)


# 훈련후 플롯.

plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()


# 할당된 클러스터별 분포


cluster_c = [len(labels[labels==i]) for i in range(cluster_count)]
cluster_n = ["Cluster "+str(i) for i in range(cluster_count)]
plt.figure(figsize=(15,5))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
plt.show()


# 분류 결과
fancy_names_for_labels = [f"Cluster {label}" for label in labels]
pd.DataFrame(zip(namesofMySeries,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")

