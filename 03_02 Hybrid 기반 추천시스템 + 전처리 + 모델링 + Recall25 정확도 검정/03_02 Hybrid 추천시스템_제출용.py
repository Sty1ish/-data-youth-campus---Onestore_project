#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 패키지 로드
import numpy as np
import scipy
import pandas as pd
import pickle
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(r'P:\새 폴더\하이브리드 추천시스템')


# In[2]:


#------------------------------------------------------------------------------------------------------------------------------
## 데이터셋 불러오기

# 충성/일반 고객 뽑기 위한 데이터셋
som_df = pd.read_csv('df_RFM_SOM_cluster.csv')

# 충성/일반 고객 ID 추출
cluster06715_df = som_df[(som_df['cluster'] == 0) | (som_df['cluster'] == 6) | (som_df['cluster'] == 7) | (som_df['cluster'] == 15)].사용자ID

# 3월 구매내역 데이터셋
purchase = pd.read_csv('원스토어_구매내역_3월.csv', index_col = 0)

# 상품사전 데이터셋
outer_goods = pd.read_pickle('outer_goods.pkl')
outer_goods1 = outer_goods.reset_index()
outer_goods1.book_name = outer_goods1.book_name.str[1:].astype(int)
outer_goods1 = outer_goods1.set_index('book_name') # book_name을 int형으로 변환

# purchase_df 생성
purchase_df = purchase.copy()
purchase_df['book_name'] = purchase_df['book_name'].str[1:].astype(int)
purchase_df = pd.merge(purchase_df, outer_goods1.reset_index(), how='left', on='book_name')
purchase_df = purchase_df.drop_duplicates(['book_name']).drop(['user_id', 'prod_id', 'buy_dt', 'hour', 'sex', 'age', 'telecom', 'buy_amt', 'category_no_y', 'dtl_category_no_y'], axis=1).reset_index(drop=True)
purchase_df.rename(columns={'category_no_x':'category_no','dtl_category_no_x':'dtl_category_no','categori_kor':'category_kor'}, inplace=True)

# 구매건수 데이터셋
df = pd.read_pickle('full_data-user_log+purchase_ratting.pkl')

# 충성/일반고객에 해당하는 사용자들의 구매건수 추출. (0건을 클릭만 한 것이므로, 1건 이상의 내역만 추출)
data = df[df.user_id.isin(cluster06715_df)].reset_index(drop=True)
data = data[data['purchase'] > 0]

# 상호작용 데이터셋 생성
interactions_df = data.copy()

# 협업 필터링을 수행할때 사용자에 대한 정보가 부족하면 cold start 문제가 발생하므로 2개 대표상품 이상의 구매이력이 있는 사용자들로만 사용
users_interactions_count_df = interactions_df.groupby(['user_id', 'book_name']).size().groupby('user_id').size()
print('사용자 수: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 2].reset_index()[['user_id']]
print('대표상품 2개 이상 구매한 사용자 수: %d' % len(users_with_enough_interactions_df))

# 2개 대표상품 이상의 구매이력이 있는 사용자들의 총 구매건수 확인
print('총 구매건수: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'user_id',
               right_on = 'user_id')
print('대표상품 2개 이상 구매한 사용자들의 총 구매건수: %d' % len(interactions_from_selected_users_df))

# 구매건수 로그변환 함수(log2)
def smooth_user_preference(x):
    return math.log(1+x, 2)

# 각 사용자/작품 별 구매지수를 선호도로 추출
interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'book_name'])['purchase'].sum().apply(smooth_user_preference).reset_index()
print(interactions_full_df.purchase.max()) # 선호도 최대값 : 9.48
print(interactions_full_df.purchase.min()) # 선호도 최소값 : 1.00

# book_name int형으로
interactions_full_df['book_name'] = interactions_full_df['book_name'].str[1:].astype(int)


# In[3]:


#--------------------------------------------------------------------------------------------------------------------------------
## 평가지표 생성

# train_test_split 80:20
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['user_id'], 
                                   test_size=0.20,
                                   random_state=42)

print('Train Dataset: %d' % len(interactions_train_df))
print('Test Dataset: %d' % len(interactions_test_df))

# personId를 Index로 설정
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')

def get_items_interacted(person_id, interactions_df):
    interacted_items = interactions_df.loc[person_id]['book_name']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

# 상호작용하지 않은 랜덤 샘플 수
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 400 

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(purchase_df['book_name'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        # Test data에서 사용자의 모든 item 가져옴
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['book_name']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['book_name'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['book_name'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        # 사용자별 추천순위 리스트
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        hits_at_25_count = 0
        
        # Test data에서 상호작용한 각각의 상품에 대해
        for item_id in person_interacted_items_testset:
            # 상호작용하지 않은 100개의 랜덤샘플 상품 추출, 이때 이 상품들은 사용자와 이전에 관련이 없었다고 가정
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            # 상호작용하지 않은 100개의 상품과 현재 상호작용하는 상품 1개 결합
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # 바로 위에서 결합한 상품 리스트를 기준으로 사용자 추천목록에 있는 상품들만 가져옴
            valid_recs_df = person_recs_df[person_recs_df['book_name'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['book_name'].values
            
            # 현재 상호작용하는 상품이 상위 N개 추천 목록에 포함되는지 확인(포함되면 1, 미포함이면 0 반환)
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10
            hit_at_25, index_at_25 = self._verify_hit_top_n(item_id, valid_recs, 25)
            hits_at_25_count += hit_at_25

        # Recall은 상호작용이 없는 상품들과 혼합될 때 상위 N개 추천 목록 중 순위가 매겨진 상호작용 상품의 비율
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
        recall_at_25 = hits_at_25_count / float(interacted_items_count_testset)

        person_metrics = {
                          'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count,
                          'hits@25_count':hits_at_25_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10,
                          'recall@25': recall_at_25
        }
        return person_metrics

    def evaluate_model(self, model):
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics)                             .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_25 = detailed_results_df['hits@25_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'recall@25': global_recall_at_25

                         }    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()


# In[4]:


#-----------------------------------------------------------------------------------------------------------------------
## 협업필터링

# 사용자별 상품에 대한 구매지수인 pivot table 생성
users_items_pivot_matrix_df = interactions_train_df.pivot(index='user_id', 
                                                          columns='book_name', 
                                                          values='purchase').fillna(0)
# pivot table에서 value값 추출
users_items_pivot_matrix = users_items_pivot_matrix_df.values
# user_id 추출
users_ids = list(users_items_pivot_matrix_df.index)
# 희소행렬이기 때문에 csr_matrix 적용
users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

# 특이값 분해 요소 수 15로 설정. 여러번 학습결과 15가 제일 적당/ 요소 수가 너무 크면 과적합되고, 너무 작으면 일반화되는 경향
NUMBER_OF_FACTORS_MF = 15
#U, sigma, Vt로 행렬 인수분해
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)
# 인수분해 후, 다시 원래 행렬로 재구성
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
# 데이터 프레임으로 전환
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False)                                     .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['book_name'].isin(items_to_ignore)]                                .sort_values('recStrength', ascending = False)                                .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'book_name', 
                                                          right_on = 'book_name')[['recStrength', 'book_name', 'book_name_kor', 'category_kor', 'goods_name_kor']]


        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, purchase_df)

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
print(cf_detailed_results_df.head(10))


# In[5]:


#--------------------------------------------------------------------------------------------------------------------
## 컨텐츠 기반 필터링

# 불용어 정의
stopword1 = pd.read_csv('korean_stopword.txt', names=['stopword'])
stopword2 = pd.read_csv('stopword.txt', names=['stopword'])
stopword = set(pd.concat([stopword1, stopword2], axis=0).stopword.values)
stopwords_list = stopword.union(set(['1부','2부', '3부', '4부', '5부', '6부', '7부', '8부', '9부', '10부', '개정판']))

# 모델 훈련
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 3),
                     min_df=0.00001,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = purchase_df['book_name'].tolist()
tfidf_matrix = vectorizer.fit_transform(purchase_df['book_name_kor'] + " " + purchase_df['category_kor'] + " " + purchase_df['goods_name_kor'] + " " + purchase_df['prod_grd_nm'])
tfidf_feature_names = vectorizer.get_feature_names()

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['book_name'])
    
    user_item_strengths = np.array(interactions_person_df['purchase']).reshape(-1,1)
    # 가중 평균 부여
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    # 정규화
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_train_df[interactions_train_df['book_name']                                                    .isin(purchase_df['book_name'])].set_index('user_id')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

# 유저 프로파일 생성
user_profiles = build_users_profiles()

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        # user profile과 item profile 사이의 코사인 유사도 계산
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # 코사인 유사도 기준으로 정렬
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['book_name', 'recStrength'])                                     .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'book_name', 
                                                          right_on = 'book_name')[['recStrength', 'book_name', 'book_name_kor', 'category_kor', 'goods_name_kor']]


        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(purchase_df)

print('Evaluating Content-Based Filtering model')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
# print(cb_detailed_results_df.head(10))


# In[6]:


#---------------------------------------------------------------------------------------------------------------------
# 하이브리드 기반 필터링
class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Top-1000 컨텐츠 기반 필터링
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        # Top-1000 협업 필터링
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        # book_name 기준으로 병합
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'outer', 
                                   left_on = 'book_name', 
                                   right_on = 'book_name').fillna(0.0)
        
        # 하이브리드 추천시스템 점수 계산
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight)                                      + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)
    
        # 하이브리드 추천시스템 점수 정렬
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'book_name', 
                                                          right_on = 'book_name')[['recStrengthHybrid', 'book_name', 'book_name_kor', 'category_kor', 'goods_name_kor']]


        return recommendations_df

# 가중치    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, purchase_df,
                                             cb_ensemble_weight=1.0, cf_ensemble_weight=500.0)

print('Evaluating Hybrid model')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
# print(hybrid_detailed_results_df.head(10))


# In[7]:


# 각 모델 별 recall@N 지수 출력
global_metrics_df = pd.DataFrame([cb_global_metrics, cf_global_metrics, hybrid_global_metrics])                         .set_index('modelName')

# plot으로 표현
ax = global_metrics_df.transpose().plot(kind='bar', figsize=(16,8))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    ax.legend(loc=2)
    
def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(purchase_df, how = 'left', 
                                                      left_on = 'book_name', 
                                                      right_on = 'book_name') \
                          .sort_values('purchase', ascending = False)[['purchase',
                                                                          'book_name',
                                                                          'book_name_kor', 'category_kor', 'goods_name_kor']]
# 어떤 사용자의 상호작용 상품 목록
print(inspect_interactions('ac0dde3b21a7991d97ec15d0360d024a', test_set=False).head(10))
# 협업필터링 추천시스템 목록
print(cf_recommender_model.recommend_items('ac0dde3b21a7991d97ec15d0360d024a', topn=10, verbose=True))
# 컨텐츠 기반 필터링 추천시스템 목록
print(content_based_recommender_model.recommend_items('ac0dde3b21a7991d97ec15d0360d024a', topn=10, verbose=True))
# 하이브리드 추천시스템 목록
print(hybrid_recommender_model.recommend_items('ac0dde3b21a7991d97ec15d0360d024a', topn=10, verbose=True))

