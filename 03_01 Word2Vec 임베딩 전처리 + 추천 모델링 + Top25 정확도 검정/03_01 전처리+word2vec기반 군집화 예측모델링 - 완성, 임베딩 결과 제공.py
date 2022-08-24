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




#%% # 전처리 과정의 설명은 03-01 베이스라인 기초 구현을 참조할것. 주석없이 최대한 짧게 진행함.
print('데이터 임포트')
# click_data
click = pd.read_csv('산학_클릭로그.csv', names = ['buy_dt', 'hour', 'user_id', 'book_name'])

# click_data dtl_category
click_dict = pd.read_csv('산학_클릭로그_상품카테고리 테이블.csv', names = ['book_name', 'category_no', 'categori_kor', 'dtl_category_no', 'goods_name_kor'])

# click_data-dict merge
click = pd.merge(click, click_dict, how = 'left', on='book_name')

# purchase data
purchase = pd.read_csv('원스토어_구매내역.csv', names = ['buy_dt', 'hour', 'user_id', 'sex',
                                                 'age', 'telecom', 'prod_id', 'book_name',
                                                 'category_no', 'dtl_category_no','prod_grd_nm','buy_amt'])

# leave only march Data - click data is march only
purchase = purchase[purchase.buy_dt.astype('str').str[4:6] == '03'].reset_index(drop = True)


# time col type (ex. "04" => int 4) encoding
purchase.hour = purchase.hour.str[:2].astype('int64')

# col select
purchase = purchase[['buy_dt','hour','user_id','book_name','dtl_category_no']]
click = click[['buy_dt','hour','user_id','book_name','dtl_category_no']]

# label append
click['purchase'] = 0; purchase['purchase'] = 1; 

print('데이터 결합시작')
full_data = pd.concat([purchase, click],axis=0)

# del variable
del click, purchase

# type change (for save memory)
full_data.buy_dt = pd.to_datetime(full_data.buy_dt, format='%Y%m%d')
full_data = full_data.astype({'user_id' : 'category', 'book_name' : 'category', 'dtl_category_no' : 'category'})


#%%
print('상품 테이블 생성.')
# goods
goods = pd.read_csv('대표상품_상세카테고리_종속상품_상품명.csv')
# Col Name change
goods = goods.drop(['prod_id', 'prod_nm'],axis = 1)
goods.columns = ['book_name', 'dtl_category_no']
# full_data book name, category extract
book_name = []
dtl_category_no = []
for idx, _ in tqdm(full_data.groupby(['book_name', 'dtl_category_no'])):
    book_name.append(idx[0])
    dtl_category_no.append(idx[1])

temp = pd.DataFrame([book_name, dtl_category_no]).T
temp.columns = ['book_name', 'dtl_category_no']
# merge dictonary
outer_goods = pd.merge(temp, goods, how='outer', on=['book_name','dtl_category_no'])

# duplicated removed
outer_goods = outer_goods.drop_duplicates(['book_name', 'dtl_category_no'], keep='first')
outer_goods = outer_goods.drop_duplicates(['book_name'], keep='first')

# merge > click_dict + duplicated removed
outer_goods = pd.merge(click_dict, outer_goods, how='outer', on=['book_name','dtl_category_no'])
outer_goods = outer_goods.drop_duplicates(['book_name'], keep='first') #앞쪽이 더 완전한 테이블이다.
# 

# memory save
del book_name, dtl_category_no, temp, goods, idx


#%%
# make ID dictionary
print('상품 테이블 문자열 라벨링.') 
# book ID mapping dict
cat = pd.read_csv('카테고리_테이블.csv', index_col = 0, names = ['categori_kor']).to_dict()
cat = cat['categori_kor']
goods_full = pd.read_csv('상품테이블.csv', index_col = 0, names = ['goods_name_kor']).to_dict()
goods_full = goods_full['goods_name_kor']
# mapping outer_goods
outer_goods['book_name_kor'] = outer_goods.book_name.map(goods_full)
outer_goods['dtl_category_no_kor'] = outer_goods.dtl_category_no.map(cat)
# set index book_name 
outer_goods = outer_goods.set_index('book_name')
# na fill unknown
outer_goods = outer_goods.fillna("unknown")
# memory save
del cat, goods_full

# 준 명단에서 매 화별 데이터가 존재해서, 이런 데이터프레임이 제작되었다.

#%%
# make dataset
print('구매 로그 테이블 작업') 
# sort data
full_data = full_data.sort_values(by=['buy_dt','hour'])

# make dataset
userid   = []
book_log = []
buy_log  = []
for idx, val in tqdm(full_data.groupby('user_id')):
    userid.append(idx)
    book_log.append(list(val.book_name))
    buy_log.append(list(val.purchase))

user_log = pd.DataFrame([userid, book_log, buy_log]).T
user_log.columns = ['user_id', 'book_log', 'buy_log']

# momory save
del userid, buy_log, book_log, idx, val, full_data


#%%
# dataset preprocessing
print('구매 로그 1인 사람, 훈련 불가 딕셔너리 제거. word to vec 전처리') 
# minimum len = 2 
short_data_idx = []
for i in tqdm(range(user_log.shape[0]), leave=True):
    if len(user_log.iloc[i].buy_log) <= 1:
        short_data_idx.append(user_log.iloc[i].user_id)

# preprocessing dataset
user_log = user_log[~user_log['user_id'].isin(short_data_idx)].reset_index(drop = True)

# 제거된 유저수
print(f'제거된 유저는 {len(short_data_idx)}명 입니다.')

#%%
# var save
with open('user_log.pkl', 'wb') as f:
    pickle.dump(user_log, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('outer_goods.pkl', 'wb') as f:
    pickle.dump(outer_goods, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
# val load
# 해당 행 이전 작업들은 이 두개 불러오면 생략가능.
with open('user_log.pkl', 'rb') as f:
    user_log = pickle.load(f)
with open('outer_goods.pkl', 'rb') as f:
    outer_goods = pickle.load(f)

del f

#%%
# word2vec callback
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss

#%%
# train_test_split = book log Word2Vec
clean_playlist = user_log.book_log

playlist_train, playlist_test = train_test_split(clean_playlist, test_size = 0.2, shuffle = True)

#%%
# train word2vec model
model = Word2Vec(window = 10, sg = 1, hs = 0, negative = 10, alpha=0.03, min_alpha=0.0007)
logging.disable(logging.NOTSET) # enable logging
t = time()
model.build_vocab(playlist_train)
logging.disable(logging.INFO) # disable logging
callback = Callback() # instead, print out loss for each epoch
t = time()

# Word2vec train train_set - epoch 50 == loss 0. << check
model.train(playlist_train,
            total_examples = model.corpus_count,
            epochs = 50,
            compute_loss = True,
            callbacks = [callback]) 

model.save("click_purchase_log2vec.model")

# 시간 로그 제거.
del t

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

# print vector (row = book, col = dim 100 embedding)
plt.figure(figsize=(20, 10)) 
sns.heatmap(X[0:10], cbar = False, cmap='PuBu', yticklabels=False)
plt.title("Bookid-word2vec embedding vector visualize",fontsize=12)
plt.show()


#%%
# tSNE 차원축소
labels = [] # book ID
tokens = [] # len 100 vector (book ID len = 100)
for word in model.wv.key_to_index:
    tokens.append(model.wv[word])
    labels.append(word)
del word

#%%
# 훈련과정 분할.
print('tSNE 차원축소 과정. 진행')
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, verbose = 1, n_jobs = -1,  random_state=23)
new_values = tsne_model.fit_transform(tokens)

# Work save
with open('new_values.pkl', 'wb') as f:
	pickle.dump(new_values, f, protocol=pickle.HIGHEST_PROTOCOL)
    

#%%    
# data load, 위 tSNE 작업 생략용. -> 대신 훈련된 모델과 같은 코드의 결과를 사용해야함. 그렇지 않으면 결과가 틀어짐.
with open('new_values.pkl', 'rb') as f:
    new_values = pickle.load(f)

#%%
# drow plot function
def tsne_plot(model, new_values, labels, print_label):
    "Creates and TSNE model and plots it"
    x = []
    y = []
    print('df 제작단계')
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    print('플롯 출력단계')
    plt.figure(figsize=(16, 16)) 
    if (print_label == 0):
        for i in tqdm(range(len(x))):
            plt.scatter(x[i],y[i])
        plt.show()
    else:
        for i in tqdm(range(len(x))):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()

#%%
def tsne_category_plot(model, new_values, labels, book_dict):
    df = pd.concat([pd.DataFrame(new_values),pd.Series(labels)],axis = 1)
    df.columns = ['x', 'y', 'book_name']
    df = pd.merge(df, book_dict.reset_index(), how='left', on='book_name')
    matplotlib.rc('font',family='gulim')
    plt.figure(figsize=(16, 16))
    matplotlib.rc('font',family='gulim')
    g = sns.scatterplot(
        x="x", y="y",
        hue='dtl_category_no',
        data=df,
        legend="full",
        alpha=0.5
        );
    g.set(xlabel=None)
    g.set(ylabel=None)
    plt.legend(loc='lower right', labelspacing=0.15, ncol=2)
    plt.show()

#%%
# drow tsne plot - working time = 15min - 비효율성으로 실행 X
# tsne_plot(model, new_values, labels, 0)

#%%
# drow plot, add label - working time = 20min - 비효율성으로 실행 X
# tsne_plot(model, new_values, labels, 1)

#%%
# drop plot category. 위 플롯 표현보다는 효율적임.
tsne_category_plot(model, new_values, labels, outer_goods)

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
# test 셋의 감상-구매 내역 기반 추천 실시.
playlist_vec = list(map(meanVectors, playlist_test))

# 유저의 이력에 따른 근처 작품 10작품을 추천하게 됨.
similarbooksByVector(playlist_vec[2], n = 12)

# 아니면 임의의 이력을 로그로 사용하여 이 추천 결과를 확인 가능하다.
similarbooksByVector(meanVectors(random.sample(list(outer_goods.index), 20)), n = 10)

# 가끔 돌리다 보면 에러 등장함. -> 훈련된 범주가 두개 이상 포함된 데이터야지만 연산이 가능함.
error_find = 0
for i in range(1024,1200):
    try:
        print(similarbooksByVector(playlist_vec[i], n = 10))
    except:
        print('not inculde case')
        error_find += 1
        continue
print(f'실행도중 {error_find}개의 에러가 발생하였습니다. = word2vec에 훈련된 값 없음.')
# 또한 책 이름이 없는 값들이 있어, unknown이 존재함. 전처리 duplicated 처리할때 생긴것루도 있다.

#%%
# 유저별 벡터로 임베딩하기
# 우리는 Word2Vec을 상품 단위로 보고 상품 추천으로도 볼 수 있지만
# 유저별로 Word2Vec을 실행한 결과값을 받을수도 있을것이다.
# 따라서, 이 결과를 구매 이력-클릭 이력으로 넣으면 어떨까 하고 생각을 해봄.

# 당연히 이 결과는 tSNE를 쓰든, PCA를 쓰든 축소된 값이 군집화 일어나야할것.
# 일단 데이터셋 부터 만들면서 생각해보자.

# 이제 길이가 다른 구매내역을 100차원에 임베딩을 시킬수 있게 되었다.
def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


#%% 임베딩 값 반환.
tokenized_docs = list(user_log['book_log'])
vectorized_docs = vectorize(tokenized_docs, model=model)
len(vectorized_docs), len(vectorized_docs[0])

# 우리는 이제 유저의 구매 내역을 임베딩한 결과를 얻게 되었다.
# 중요한 사실은 이제 클릭, 구매를 1번 이하로 한 사람은 군집화 할 필요성을 못느꼈다는 가정이 필요하다.

vectorized_docs[:5]


# 이 변수를 유저별로 추가해주면 해결될것.
w2v_df = pd.concat([user_log.user_id.to_frame(), pd.DataFrame(vectorized_docs, columns = ['w2v_'+str(i) for i in range(1,101)])], axis = 1)

with open('buy+click_word2vec_embedding.pkl', 'wb') as f:
    pickle.dump(w2v_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('buy+click_word2vec_embedding.pkl', 'rb') as f:
    w2v_df = pickle.load(f)

del tokenized_docs, vectorized_docs, w2v_df

#%%
# 추천 성능 측정.
# 소요시간 40시간 근처.
# 앵간하면 실행하지 맙시다.

# 로그 기준 25개의 책을 추천한다.
top_n_books = 25

#
# 랜덤 추천,
#
# 유저의 이력 데이터중 하나를 빼고, 리스트로 25개를 추천한뒤, 그 하나가 추천 목록에 포함되면1, 아니면 0을 반환.
# 소요시간 3-6시간.
def hitRateRandom(playlist, n_books):
    hit = 0
    for i, target in enumerate(playlist):
        random.seed(i)
        recommended_books = random.sample(list(outer_goods.index), n_books)
        hit += int(target in recommended_books)
    return hit/len(playlist)

eval_random = pd.Series([hitRateRandom(p, n_books = top_n_books) for p in tqdm(playlist_test)])
# top 25 ACC
eval_random.mean()


#
# 태그 기준 추천.
#
# 유저가 감상한 태그에 해당하는 책 제목만 가져옵니다. 그 리스트중 하나를 랜덤하게 추천하게 됩니다.
# 실행시 에러메시지가 엄청 등장합니다. 주의바람. 소요시간 3-6시간.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

mapping_tag2book = outer_goods.explode('dtl_category_no').reset_index().groupby('dtl_category_no')['book_name'].apply(list)

def hitRateBookTag(playlist, window, n_books):
    hit = 0
    context_target_list = [([w for w in range(idx-window, idx+window+1) if not(w < 0 or w == idx or w >= len(playlist))], target) for idx, target in enumerate(playlist)]
    
    for i, (context, target) in enumerate(context_target_list):
        # 나는 index가 0~... 이 아니라 책 제목이니까 이렇게 잡아야지.
        context_book_tags = set(outer_goods.iloc[context,:].loc[:,'dtl_category_no'].explode().values)
        possible_books_id = set(mapping_tag2book[context_book_tags].explode().values)
        
        random.seed(i)
        recommended_books = random.sample(possible_books_id, n_books)
        hit += int(target in recommended_books)
        
    return hit/len(playlist)

eval_book_tag = pd.Series([hitRateBookTag(p, model.window, n_books = top_n_books) for p in tqdm(playlist_test, position=0, leave=False)])

# Top 25 ACC
eval_book_tag.mean()
 
   
#
# Word2Vec 추천
#
# 유저의 구매 이력을 바탕으로 주변 노래의 평균벡터를 가저온다.
# 위 함수를 이용해 코사인 유사도를 기반으로 상위 25개의 유사한 노래를 추천합니다. 소요시간 약 2-3시간.
# 
def hitRatelog2Vec(playlist, window, n_books):
    hit = 0
    context_target_list = [([w for w in range(idx-window, idx+window+1) if not(w < 0 or w == idx or w >= len(playlist))], target) for idx, target in enumerate(playlist)]
    for context, target in context_target_list:
        context_vector = meanVectors(context)
        recommended_books = similarbooksByVector(context_vector, n = n_books, by_name = False)
        songs_id = list(zip(*recommended_books))[0]
        hit += int(target in songs_id)
    return hit/len(playlist)

eval_log2vec = pd.Series([hitRatelog2Vec(p, model.window, n_books = top_n_books) for p in tqdm(playlist_test, position=0, leave=True)])
eval_log2vec.mean()


#
# 각 추천별 top25 내 정확도.
#
print(f'top 25 acc 랜덤 예측 정확도 :  {eval_random.mean()}')
print(f'top 25 acc 태그 기준 예측 정확도 :  {eval_book_tag.mean()}')
print(f'top 25 acc log2vec 유사도 기준 예측 정확도 :  {eval_log2vec.mean()}')

# 평가 결과 저장. - 너무 오래걸려.
# eval_random.mean()
# Out[16]: 0.00010147498646098818
# word2vec 추천했을때 ACC가 0.006291...으로 등장했음. 코드 다돌고 나서 확인.



# top 25 acc 랜덤 예측 정확도 :  0.00010147498646098818
# top 25 acc 태그 기준 예측 정확도 :  6.917034273518259e-05
# top 25 acc log2vec 유사도 기준 예측 정확도 :  0.006291165588886587

# 추가로 03-04에서 진행한, top 25 acc DNN 모델 예측 정확도 : Out[1]: 0.03723943661971831

plot_data = pd.DataFrame([['random top25 recomend', eval_random.mean()],
              ['cat based top25 recomend', eval_book_tag.mean()],
              ['log2vec based top25 recomend', eval_log2vec.mean()]],
             columns = ['names', 'val'])



# 정답 비율 어떻게 되는지, 
sns.barplot(x= 'names', y = 'val', data = plot_data)
plt.title('Top25 ACC by Recomend method')
plt.show()



#%%
# 재생 가능하게 플롯 그리기

# top 25 acc 랜덤 예측 정확도 :  0.00010147498646098818
# top 25 acc 태그 기준 예측 정확도 :  6.917034273518259e-05
# top 25 acc log2vec 유사도 기준 예측 정확도 :  0.006291165588886587

# 추가로 03-04에서 진행한, top 25 acc DNN 모델 예측 정확도 : Out[1]: 0.03723943661971831

plot_data = pd.DataFrame([['cat based top25', 6.917034273518259e-05],
                          ['random top25', 0.00010147498646098818],
                          ['log2vec based top25', 0.006291165588886587],
                          ['DNN model top25', 0.03723943661971831]],
                         columns = ['names', 'val'])


ax = sns.barplot(x= 'names', y = 'val', data = plot_data)
tb = round(plot_data.val,6)
ax.bar_label(ax.containers[0], labels=tb, padding=3)
plt.title('Top25 ACC by Recomend method')
plt.show()

#%%



