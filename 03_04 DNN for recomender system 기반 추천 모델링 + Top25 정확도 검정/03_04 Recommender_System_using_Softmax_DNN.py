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

# Modelling -> gensim은 기존결과랑 비교를 위해서.
from scipy import stats
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE


# NN
import tensorflow as tf
import keras
from pprint import pprint
from keras.callbacks import EarlyStopping

# Additional
import os
import math
import random
from tqdm import tqdm
import pickle



os.chdir(r'C:\Users\9001283\Desktop\03-01 Word2Vec 기반 군집화 - 작품추천')



#%%
# 전처리 과정의 설명은 03-01 베이스라인 기초 구현을 참조할것. 주석없이 최대한 짧게 진행함.
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
full_data = full_data.drop(['hour','buy_dt'],axis = 1)
# 03-02의 user_log를 이용하지 않고, 이 모델은 full_data를 이용한다.


list_user_id = []
list_book_name = []
list_dtl_category_no = []
list_purchase = []
# 우리는 03-02의 학습법이 아니라, 유저가 특정책을 몇번 구매했냐를 점수로 준다.
for idx, val in tqdm(full_data.groupby(['user_id', 'book_name', 'dtl_category_no'])):
    list_user_id.append(idx[0])
    list_book_name.append(idx[1])
    list_dtl_category_no.append(idx[2])
    list_purchase.append(int(val.purchase.sum()))

# full_data 재정의.
full_data = pd.DataFrame({'user_id': list_user_id, 'book_name':list_book_name, 'dtl_category_no':list_dtl_category_no, 'purchase':list_purchase})

with open('full_data-user_log+purchase_ratting.pkl', 'wb') as f:
    pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)

del list_user_id, list_book_name, list_dtl_category_no, list_purchase

#%%    
with open('full_data-user_log+purchase_ratting.pkl', 'rb') as f:
    full_data = pickle.load(f)

#%%
# 딕셔너리는 전체를 사용하지 않고 훈련에 등장한 녀석들만 사용한다.
outer_goods = outer_goods[list(outer_goods.reset_index().book_name.isin(full_data.book_name))]

# purchase 변수 가정으로, 구매가 평점이 높다, 클릭은 평점이 낮다 가정. -> 클릭만 많이하면 아래 작품으로, 구매 많이하면 그나마 높은 값으로 학습될것이기에.
# rating이라는 변수를 purchase로 대신 보았다.

user_lab = LabelEncoder()
full_data['user'] = user_lab.fit_transform(full_data['user_id'].values)
n_users = full_data['user'].nunique()

book_lab = LabelEncoder()
book_lab.fit(outer_goods.index)
full_data['book'] = book_lab.transform(full_data['book_name'].values)
n_books = outer_goods.index.nunique()

cat_lab = LabelEncoder()
cat_lab.fit(outer_goods.dtl_category_no)
full_data['category'] = cat_lab.transform(full_data['dtl_category_no'].values)
n_categorys = outer_goods.dtl_category_no.nunique()

full_data['rating'] = full_data['purchase'].values.astype(np.float32)
min_rating = min(full_data['rating'])
max_rating = max(full_data['rating'])

# 

print(f'고유 유저수 : {n_users}')
print(f'고유 책수 : {n_books}')
print(f'고유 카테고리수 : {n_categorys}')
print(f'구매수 최소, 최대 : {min_rating, max_rating}')

# 원래대로 돌리려면 이렇게 해야지. 
# user_enc.inverse_transform(full_data.user_id)

#%%

X = full_data[['user', 'book','category']].values
y = full_data['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=404)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=404)

print('X train valid test shape & y same')
print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)



# input shape 변경.
X_train_array = [X_train[:, 0], X_train[:, 1], X_train[:, 2]]
X_valid_array = [X_valid[:, 0], X_valid[:, 1], X_valid[:, 2]]
X_test_array = [X_test[:, 0], X_test[:, 1], X_test[:, 2]]

X_train, X_train_array, X_train_array[0].shape
X_train_array[0].shape, X_train_array[1].shape, X_train_array[2].shape
# Scaling -> not need = 원래 0 혹은 1뿐인 데이터임.
# y_train = (y_train - min_rating)/(max_rating - min_rating)
# y_test = (y_test - min_rating)/(max_rating - min_rating)

#%%

# 레이어의 임베딩 차원 150으로 설정.
n_factors = 150

# 모델 생성.
## Initializing a input layer for users
user_layer = tf.keras.layers.Input(shape = (1,))

## Embedding layer for n_factors of users
u = tf.keras.layers.Embedding(n_users, n_factors, embeddings_initializer = 'he_normal',embeddings_regularizer = tf.keras.regularizers.l2(1e-6))(user_layer)
u = tf.keras.layers.Reshape((n_factors,))(u)

## Initializing a input layer for movies
book_layer = tf.keras.layers.Input(shape = (1,))

## Embedding layer for n_factors of movies
m = tf.keras.layers.Embedding(n_books, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(book_layer)
m = tf.keras.layers.Reshape((n_factors,))(m)

## Initializing a input layer for movies
category_layer = tf.keras.layers.Input(shape = (1,))

## Embedding layer for n_factors of movies
ct = tf.keras.layers.Embedding(n_categorys, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(category_layer)
ct = tf.keras.layers.Reshape((n_factors,))(ct)

## stacking up both user and movie embeddings
x = tf.keras.layers.Concatenate()([u,m,ct])
x = tf.keras.layers.Dropout(0.05)(x)

## Adding a Dense layer to the architecture
x = tf.keras.layers.Dense(32, kernel_initializer='he_normal')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dropout(0.05)(x)

x = tf.keras.layers.Dense(16, kernel_initializer='he_normal')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dropout(0.05)(x)

## Adding an Output layer with Sigmoid activation funtion which gives output between 0 and 1
# 출력층은 1개여야함. 또한 항등함수여야함. 회귀 예측값이어야하기 때문.
x = tf.keras.layers.Dense(1)(x)
# x = tf.keras.layers.Activation(activation='softmax')(x)

## Adding a Lambda layer to convert the output to rating by scaling it with the help of available rating information
# x = tf.keras.layers.Lambda(lambda x: x*(max_rating - min_rating) + min_rating)(x)

## Defining the model
model = tf.keras.models.Model(inputs=[user_layer, book_layer, category_layer], outputs=x)
# optimizer = tf.keras.optimizers.Adam(lr=0.001)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005,
    # rho=0.9, momentum=0.01, epsilon=1e-07)

## Compiling the model
# model.compile(loss='binary_crossentropy', optimizer = optimizer)
# model.compile(loss='mean_squared_error', optimizer = optimizer,metrics=['accuracy'])
# 속도가 우선이니 rmsprop 선정, mse를 loss와 매개변수로 지정.
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_squared_error'])


#%%

model.summary()

#%%
early_stopping = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=0.000001, verbose=1)

# 훈련. 15시간 소요.
history = model.fit(x = X_train_array, y = y_train, batch_size=128, epochs=5, verbose=1,
                    validation_data=(X_valid_array, y_valid) ,shuffle=True,callbacks=[reduce_lr, early_stopping])


#%%
model.save('model_DNN_epoch5.h5')

#%%
model = tf.keras.models.load_model('model_DNN_epoch5.h5')

#%%
# 학습률 곡선

plt.plot(history.history["loss"][5:])
plt.plot(history.history["val_loss"][5:])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

#%%

# 누구 한명의 결과를 보고싶다면 이 코드를 실행시킬것.
# 추천 알고리즘 돌아가는 방법.
# 유저, 책, 카테고리는 사전 fit이 완료되어야함. model은 인풋 받아야함. outer_goods 데이터셋, full_data가 존재해야함.
user_id = [4115] # 랜덤으로 변경해볼것.


encoded_user_id = user_lab.inverse_transform(user_id)
print(f'유저명 : {encoded_user_id}')

# 본 책과 카테고리.
seen_books = list(full_data[full_data['user'] == user_id[0]]['book'])
seen_cat = outer_goods[outer_goods.index.isin(book_lab.inverse_transform(seen_books))].dtl_category_no.unique()

# 안본 책들 + 카테고리
unseen_book = outer_goods[~outer_goods.index.isin(book_lab.inverse_transform(seen_books))]
unseen_book = unseen_book[unseen_book.dtl_category_no.isin(seen_cat)]

# 다시 라벨 인코딩.
unseen_books = unseen_book.index
unseen_books = book_lab.transform(unseen_books)
unseen_books_cat = unseen_book.dtl_category_no
unseen_books_cat = cat_lab.transform(unseen_books_cat)

#모델 input shape 조작.
model_input = [np.asarray(list(user_id)*len(unseen_books)), np.asarray(unseen_books), np.asarray(unseen_books_cat)]
len(model_input), len(model_input[0]), len(model_input[1]), len(model_input[2])  

# 추천 리스트 만들기.
predicted_ratings = model.predict(model_input)
predicted_ratings = pd.concat([pd.Series(unseen_books), pd.Series(predicted_ratings.reshape(-1,))], axis=1)
predicted_ratings.columns = ['unseen_books', 'predict_score']
# 점수별로 정렬하기.
predicted_ratings = predicted_ratings.sort_values('predict_score', ascending = False)

# print(sorted_index)
recommended_books = book_lab.inverse_transform(predicted_ratings.unseen_books)
recommended_books
# 매핑
recommended_books_df = pd.DataFrame([recommended_books,recommended_books]).T
recommended_books_df.columns = ['book_name', 'book_name_kor']
recommended_books_df.book_name_kor = recommended_books_df['book_name_kor'].map(outer_goods.loc[:,'book_name_kor'].to_dict()).to_frame()

# unknown 제거. > 추천목록.
print('추천 상위 25개')
print(recommended_books_df[recommended_books_df.book_name_kor != 'unknown'].head(25))
print('유저 이력')
print(outer_goods[list(outer_goods.reset_index().book_name.isin(list(book_lab.inverse_transform(seen_books))))])





#%%
# 개념상 코딩해봐야지.
# 일단 기본적으로 한명에 대해서 볼 작품 예상하는법.

# 우리는 훈련에 사용하지 않은. test 데이터셋 이정도를 들고 있다. (전체 20%)
# model_input = X_test_array 이고. [int(X_test_array[0][0])] [int(X_test_array[1][0])], [int(X_test_array[2][0])] 로 불러와야함.
# int(X_test_array[0][0])
# y_test 


# input을 user_번호를 받는다. int(X_test_array[0][1])이 인풋이 된다.
def DNN_top25recommender(X_test_user, head_num = 25):
    user_id = X_test_user
    
    # 데이터셋 제작.
    seen_books = list(full_data[full_data['user'] == user_id]['book'])
    
    # 테스트 하나 제외.
    chk_idx = random.randint(0, len(seen_books)-1)
    test_book = seen_books[chk_idx]
    test_book_unlab = book_lab.inverse_transform([test_book])
    seen_books.remove(test_book)
    
    # 카테고리 생성.
    seen_cat = outer_goods[outer_goods.index.isin(book_lab.inverse_transform(seen_books))].dtl_category_no.unique()
    
    # 조기종료.
    if len(seen_books) <= 0: # 한건이면 지웠을때 테스트가 불가능하지.
        return '측정불가'
    
    # 안본 책들 + 카테고리
    unseen_book = outer_goods[~outer_goods.index.isin(book_lab.inverse_transform(seen_books))]
    unseen_book = unseen_book[unseen_book.dtl_category_no.isin(seen_cat)]

    # 다시 라벨 인코딩.
    unseen_books = unseen_book.index
    unseen_books = book_lab.transform(unseen_books)
    unseen_books_cat = unseen_book.dtl_category_no
    unseen_books_cat = cat_lab.transform(unseen_books_cat)

    #모델 input shape 조작.
    model_input = [np.asarray(list([user_id])*len(unseen_books)), np.asarray(unseen_books), np.asarray(unseen_books_cat)]
    
    # 추천 리스트 만들기.
    predicted_ratings = model.predict(model_input)
    predicted_ratings = pd.concat([pd.Series(unseen_books), pd.Series(predicted_ratings.reshape(-1,))], axis=1)
    predicted_ratings.columns = ['unseen_books', 'predict_score']
    # 점수별로 정렬하기.
    predicted_ratings = predicted_ratings.sort_values('predict_score', ascending = False)

    recommended_books = book_lab.inverse_transform(predicted_ratings.unseen_books)
    recommended_books
    # 매핑
    recommended_books_df = pd.DataFrame([recommended_books,recommended_books]).T
    recommended_books_df.columns = ['book_name', 'book_name_kor']
    recommended_books_df.book_name_kor = recommended_books_df['book_name_kor'].map(outer_goods.loc[:,'book_name_kor'].to_dict()).to_frame()
    
    recommended_books_df = recommended_books_df.head(head_num)
    
    # 추천목록에 존재하는지 안하는지 반환. 당연히 책은 하나만 존재하므로 0-1 반환.
    return recommended_books_df.book_name.isin(test_book_unlab).sum()

#%%
# Epoch 1/5
# 14072/14072 [==============================] - 7208s 512ms/step - loss: 26.6288 - mean_squared_error: 26.6277
# - val_loss: 25.4363 - val_mean_squared_error: 25.4340 - lr: 0.0010

# 원래라면 전부 실험해봐야 할것이나, 730시간 예상으로(일정내 훈련불가.) 64만건중 10%인 6.4만건만 진행하였다.
# train_test_split에서 시드 고정이라서, user_id 순서는 같게 나옴. -> 훈련 연속하여 진행.
# 계속 훈련이 터져서, 해당번째부터 이어서 진행하는 식으로 진행.
test_lists = []
counter = 26000
for user_ids in tqdm(X_test_array[0][26000:], leave=False):
    test_lists.append(DNN_top25recommender(user_ids))
    counter += 1
    # 하도 많이 터져서, 중간중간에 세이브 포인트 만듬.
    if counter % 500 == 0:
        with open('test_list_from+26000_to'+str(counter)+'length_test.pkl', 'wb') as f:
            pickle.dump(test_lists, f, protocol=pickle.HIGHEST_PROTOCOL)

# top25정확도
np.mean(test_lists)



#%%    
# 콘솔 다른것에서 켜서 중간점검 해보자.
import os
import numpy as np
import pickle
os.chdir(r'C:\Users\9001283\Desktop\03-01 Word2Vec 기반 군집화 - 작품추천')
# 테스트 끝난 개수 들고오기. - predict가 계속 잘려서 진행되서, 여러개 파일에 분할되서 저장됨.
with open('[test_set-important]_test_list9000length_test.pkl', 'rb') as f:
    test_lists = pickle.load(f)

# 2회차 테스트.
with open('[test_set-important]_test_list_from+9000_to18000length_test.pkl', 'rb') as f:
    test_lists2 = pickle.load(f)

test_lists.extend(test_lists2)
del test_lists2

# 3회차 테스트.
with open('[test_set-important]_test_list_from+18000_to26000length_test.pkl', 'rb') as f:
    test_lists3 = pickle.load(f)

test_lists.extend(test_lists3)
del test_lists3

with open('[test_set-important]_test_list_from+26000_to35500length_test.pkl', 'rb') as f:
    test_lists4 = pickle.load(f)

test_lists.extend(test_lists4)
del test_lists4



# 측정 불가건은 추천 불가능, 오답으로 평가한다고 가정할때.
while True:
    try:
        test_lists.remove('측정불가')
        test_lists.append(0)
    except:
        break

np.mean(test_lists)
