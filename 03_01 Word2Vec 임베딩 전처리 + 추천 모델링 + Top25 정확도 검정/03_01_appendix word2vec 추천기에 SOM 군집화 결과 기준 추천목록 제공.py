# this cord version => scikit_leran 0.22 need
# from spherecluster import SphericalKMeans

# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')
sns.set_style("whitegrid")

# Modelling
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy import stats

# Additional
import os
import math
import random
import itertools
import multiprocessing
from tqdm import tqdm
from time import time
import logging
import pickle
from datetime import datetime, timedelta


os.chdir(r'C:\Users\9001283\Desktop\03-01 Word2Vec 기반 군집화 - 작품추천')



#%%
# val load
# 해당 행 이전 작업들은 이 두개 불러오면 생략가능.
with open('user_log.pkl', 'rb') as f:
    user_log = pickle.load(f)
with open('outer_goods.pkl', 'rb') as f:
    outer_goods = pickle.load(f)

del f



#%%
# OR load model
# 모델 훈련과정 스킵가능. 
model = Word2Vec.load("click_purchase_log2vec.model")

#%%
print(model)

# models vector(weight)
X = model.wv.get_normed_vectors() 
print(X)

# models vector(weight) shape
print(X.shape)


#%%
# BookID - near Book recommend function
# 특정한 유저가 감상한 순서대로 BookID의 리스트를 변수로 받습니다.
# 등장하지 않은 범주는, 무시합니다.
def meanVectors(playlist):
    vec = []
    for song_id in playlist:
        try:
            vec.append(model.wv[song_id])
        except KeyError:
            continue
    return np.mean(vec, axis=0)

# 코사인 거리 기준 상위 n개의 유사한 노래 추천
def similarbooksByVector(vec, n = 10, by_name = True):
    # extract most similar songs for the input vector
    similar_books = model.wv.similar_by_vector(vec, topn = n)
    
    # extract name and similarity score of the similar products
    if by_name:
        similar_books = [(outer_goods.loc[book_id, "book_name_kor"], sim) for book_id, sim in similar_books]
    
    return similar_books

#%%
# 클러스터에 맞는거만 결과를 보자.
clt_id = pd.read_csv('02-03-df_cluster_임베딩포함.csv')
# 클러스터별 결과보기.
# '02-03-df_cluster_임베딩X.csv' 이름만 보면됨.
# '02-03-df_cluster_임베딩포함.csv'

clt_list = list(clt_id[clt_id.cluster == 8].사용자ID)

playlist_test = user_log[user_log.user_id.isin(clt_list)].book_log

# test 셋의 감상-구매 내역 기반 추천 실시.
playlist_vec = list(map(meanVectors, playlist_test))


# 가끔 돌리다 보면 에러 등장함. -> 훈련된 범주가 두개 이상 포함된 데이터야지만 연산이 가능함.
error_find = 0
for i in range(100,150):
    try:
        print(similarbooksByVector(playlist_vec[i], n = 10))
    except:
        print('not inculde case')
        error_find += 1
        continue

print(f'실행도중 {error_find}개의 에러가 발생하였습니다. = word2vec에 훈련된 값 없음.')
# 또한 책 이름이 없는 값들이 있어, unknown이 존재함. 전처리 duplicated 처리할때 생긴것루도 있다.
