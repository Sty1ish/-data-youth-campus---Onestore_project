# baseline
# https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-transformer.ipynb
# neural collaborate filtering을 진행한것. -> 제일 구조가 간단.

# DIN의 문제가 해결된게 Transformer model이지.
# 근데 이렇게 예측하면, 구매인지, 클릭인지 판단하는 모델일 뿐인데?
# 이건 예측의 의미가 없지 않나?

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

# python modules
import os
import math
import random
import itertools
import multiprocessing
from tqdm import tqdm
from time import time
import logging
import pickle
import warnings

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy import stats
from tabulate import tabulate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
warnings.simplefilter("ignore")

os.chdir(r'C:\Users\9001283\Desktop\03-01 Word2Vec 기반 군집화 - 작품추천')
print("Keras version " + tf.keras.__version__)
print("Tensorflow version " + tf.__version__)

#%%
# data_import

# 그러고 보니, 클릭 아이디에 유저 정보가 없어서. 이상적인 형태가 구성이 안되는구나.
# 유저ID / 구매 이력 / 별점 / 유저성별 / 나이 / 분류 세트를 원래 데이터셋에서는 사용했다.
# 모델 구성시 click에 유저이력 받으면 더 잘 만들수 있을것 같지만, 그냥 데이터셋 그대로 사용하는 식으로 진행하였다.

# 03-02 Dataset을 그대로 사용한다.
with open('user_log.pkl', 'rb') as f:
    user_log = pickle.load(f)

# 사실 사용할 일이 없을수도 있다.
with open('outer_goods.pkl', 'rb') as f:
    outer_goods = pickle.load(f)

del f

# 구매-클릭이력이 몇개인지 확인.
user_log.book_log.apply(lambda x: len(x)).hist(bins = 100)

# 50% 지점이 얼마인지 확인. 20이 최종 지점이다. 그러면 최근 20개의 값에 대해 이력으로 삼으면 되겠지.
#  > baseline=96임 그래도 클릭-구매 로그 섞였으니 20으로 함.
user_log.book_log.apply(lambda x: len(x)).quantile(q=0.5, interpolation='nearest')

#%%
# 이단계에, 전처리가 필요하다. sequence_length보다 짧은 데이터가 있기 때문.
# 훈련에 8개 미만인 애들은 들고오지 않거나, 여기서 제로필을 하고 진행해야함.
# 제로필을 생각했는데, 일단 제외를 먼저 시켜보기로 함. 제로필은 어떤 역효과가 날지 판단이 안서서.
# 이거 어떤 샘플링인지 확인할것 < 이해 잘 안됨.

print(f'훈련 샘플에서 {(user_log.book_log.apply(lambda x: len(x)) < 8).sum()}건을 제거하였습니다.')
user_log = user_log[user_log.book_log.apply(lambda x: len(x)) >= 8]


#%%

sequence_length = 8
step_size = 1

def create_sequences(values, window_size, step_size):    

    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


user_log.book_log = user_log.book_log.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

user_log.buy_log = user_log.buy_log.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)


# 데이터 확장.
user_data_books = user_log[["user_id", "book_log"]].explode("book_log", ignore_index=True)
user_data_buys = user_log[["buy_log"]].explode("buy_log", ignore_index=True)


user_log_transformed = pd.concat([user_data_books, user_data_buys], axis=1)

# 이작업은 유저 로그에 대한 정보가 없으므로 필요가 없다. 무시.
# 혹시 존재한다면, 여기서 user_id에 대해 유저 정보를 조인해줄것,
# user_log_transformed = user_log_transformed.join(users.set_index("user_id"), on="user_id")

user_log_transformed.book_log = user_log_transformed.book_log.apply(lambda x: ",".join(x))

# 문자열 처리 시켜준다. 원래 형태였던 rating은 float형이라서 문자열로 바꿔준거, 우리 구매-판매도 int라서 전환.
user_log_transformed.buy_log = user_log_transformed.buy_log.apply(lambda x: ",".join([str(v) for v in x]))

# 열이름 새로 구성한다.
user_log_transformed.rename(
    columns={"book_log": "sequence_book_log", "buy_log": "sequence_buy_log"},
    inplace=True,
)




#%%
# train_test_split 
train_data, test_data = train_test_split(user_log_transformed, test_size = 0.2, shuffle = True)

# input의 행을 섞어서 input받기 위해 파일로 저장한다.
train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
test_data.to_csv("test_data.csv", index=False, sep="|", header=False)





#%%


CSV_HEADER = list(user_log_transformed.columns)
USER_FEATURES = [] # 이번 데이터 구성에는 존재하지 않음.
MOVIE_FEATURES = ["category_no, dtl_category_no"] # movies의 장르 열은 "Action|Crime|Thriller" 식으로 구성이 되어있다.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "user_id": list(user_log.user_id.unique()),
    "books_log": list(outer_goods.index.unique()),
    "category_no": list(outer_goods.category_no.unique()),
    "dtl_category_no": list(outer_goods.dtl_category_no.unique()),
}


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    def process(features):
        book_ids_string = features["sequence_book_log"]
        sequence_book_log = tf.strings.split(book_ids_string, ",").to_tensor()

        # The last movie id in the sequence is the target movie
        features["target_book_id"] = sequence_book_log[:, -1]
        features["sequence_book_log"] = sequence_book_log[:, :-1]

        ratings_string = features["sequence_buy_log"]
        sequence_buy_log = tf.strings.to_number(
            tf.strings.split(ratings_string, ","), tf.dtypes.float32
        ).to_tensor()

        # The last rating in the sequence is the target for the model to predict
        target = sequence_buy_log[:, -1]
        features["sequence_buy_log"] = sequence_buy_log[:, :-1]

        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        field_delim="|",
        shuffle=shuffle,
    ).map(process)

    return dataset


#%%
def create_model_inputs():
    return {
        "user_id": layers.Input(name="user_id", shape=(1,), dtype=tf.string),
        "sequence_book_log": layers.Input(
            name="sequence_book_log", shape=(sequence_length - 1,), dtype=tf.string
        ),
        "target_book_id": layers.Input(
            name="target_book_id", shape=(1,), dtype=tf.string
        ),
        "sequence_buy_log": layers.Input(
            name="sequence_buy_log", shape=(sequence_length - 1,), dtype=tf.float32
        ),
    }



#%%

# 코드 치환시에
# BST: Feature Encoding 부터 코딩해볼것. / 이 아래부터.
# rating_data == user_log 이다. / 
# ratings_data_transformed == user_log_transformed <열이름> {"book_log": "sequence_book_log", "ratings": "sequence_buy_log"}
# ratings_data_movies == user_data_books 이다. / ratings_data_rating == user_data_buys
# user_id는 그대로, "movies_id"=="books_log" , "ratings" == "buy_log"를 사용하였다.
# movies == outer_goods그대로 보면되고, 
# MOVIE_FEATURES에 장르 대신 ["category_no, dtl_category_no"]이 들어갔다.

'''
users, ratings, movies 데이터를 받았고, 각 셋에는 
names=["user_id", "sex", "age_group", "occupation", "zip_code"],
names=["user_id", "movie_id", "rating", "unix_timestamp"],
names=["movie_id", "title", "genres"],

USER_FEATURES = [] # 이번 데이터 구성에는 존재하지 않음.
MOVIE_FEATURES = ["category_no, dtl_category_no"] 변수를 만들기도 했었음.

  include_user_id=True,
  include_user_features=True,
  include_movie_features=True,
이런 형태로 구성되어 있다.
'''
  
#%%

def encode_input_features(
    inputs,
    include_user_id=True,
    include_user_features=True,
    include_movie_features=True,
):

    encoded_transformer_features = []
    encoded_other_features = []
    other_feature_names = []
    
    if include_user_id: # 거의 당연히 있어야 할 부분.
        other_feature_names.append("user_id")
        
    if include_user_features:
        other_feature_names.extend(USER_FEATURES)  # 우리 USER_FEATURES 값은 []로 들어가있다. 없으니까. 

    ## Encode user features
    for feature_name in other_feature_names:
        # Convert the string input values into integer indices.
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        idx = StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)(
            inputs[feature_name]
        )
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f"{feature_name}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder(idx))

    ## Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    ## Create a movie embedding encoder
    books_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY["books_log"]
    books_embedding_dims = int(math.sqrt(len(books_vocabulary)))
    # Create a lookup to convert string values to integer indices.
    books_index_lookup = StringLookup(
        vocabulary=books_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name="books_index_lookup",
    )
    # Create an embedding layer with the specified dimensions.
    books_embedding_encoder = layers.Embedding(
        input_dim=len(books_vocabulary),
        output_dim=books_embedding_dims,
        name=f"books_embedding",
    )
    
    # Create a vector lookup for movie genres. > 우리는 장르 대신에, 대표장르(4종)/44개 두개 인풋이 됨.
    # 즉 for문 구성이 위처럼 여기서도 되야한다는걸 의미하겠지? 근데 원본의 인풋이 벡터형이니까, 이처럼 변형하자.
    cat_vectors = pd.concat([pd.get_dummies(outer_goods['category_no']), pd.get_dummies(outer_goods['dtl_category_no'])],axis = 1)
    cat_vectors = cat_vectors.to_numpy()
    books_cat_lookup = layers.Embedding(
        input_dim=cat_vectors.shape[0],
        output_dim=cat_vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(cat_vectors),
        trainable=False,
        name="cat_vectors",
    )

    # Create a processing layer for genres.
    books_embedding_processor = layers.Dense(
        units=books_embedding_dims,
        activation="relu",
        name="process_movie_embedding_with_genres",
    )
    
    ############ 여기 두줄 수정이 핵심. 그에 맞춰서 아래도 바껴야할거고.

    ## Define a function to encode a given movie id.
    def encode_book(books_id):
        # Convert the string input values into integer indices.
        books_idx = books_index_lookup(books_id)
        books_embedding = books_embedding_encoder(books_idx)
        encoded_books = books_embedding
        if include_movie_features:
            books_cat_vector = books_cat_lookup(books_idx)
            encoded_books = books_embedding_processor(
                layers.concatenate([books_embedding, books_cat_vector])
            )
        return encoded_books

    ## Encoding target_movie_id
    target_books_id = inputs["target_book_id"]
    encoded_target_books = encode_book(target_books_id)

    ## Encoding sequence movie_ids.
    sequence_books_ids = inputs["sequence_book_log"]
    encoded_sequence_books = encode_book(sequence_books_ids)
    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=sequence_length,
        output_dim=books_embedding_dims,
        name="position_embedding",
    )
    
    positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # Retrieve sequence ratings to incorporate them into the encoding of the movie.
    sequence_buy_log = tf.expand_dims(inputs["sequence_buy_log"], -1)
    # Add the positional encoding to the movie encodings and multiply them by rating.
    encoded_sequence_movies_with_poistion_and_rating = layers.Multiply()(
        [(encoded_sequence_books + encodded_positions), sequence_buy_log]
    )

    # Construct the transformer inputs.
    for encoded_movie in tf.unstack(
        encoded_sequence_movies_with_poistion_and_rating, axis=1
    ):
        encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
    
    encoded_transformer_features.append(encoded_target_books)
    encoded_transformer_features = layers.concatenate(encoded_transformer_features, axis=1)

    return encoded_transformer_features, encoded_other_features



#%%
# Transfomer model 구조
# 애초에 이건 위에 매개변수 위에 건들 필요가 없다.

include_user_id = False
include_user_features = False
include_movie_features = False

hidden_units = [256, 128]
dropout_rate = 0.1
num_heads = 3


def create_model():
    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(
        inputs, include_user_id, include_user_features, include_movie_features
    )

    # Create a multi-headed attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other features.
    if other_features is not None:
        features = layers.concatenate(
            [features, layers.Reshape([other_features.shape[-1]])(other_features)]
        )

    # Fully-connected layers.
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    # 우리는 점수값을 리턴받는게 아닌, 0인지 1인지를 받고 싶다.
    outputs = layers.Dense(units=2)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

#%%
# 모델생성, 훈련, 

model = create_model()

model.compile(
    optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

# Read the training data.
train_dataset = get_dataset_from_csv("train_data.csv", shuffle=True, batch_size=265)

# Fit the model with the training data.
model.fit(train_dataset, epochs=1, verbose = 1)

# Read the test data.
test_dataset = get_dataset_from_csv("test_data.csv", batch_size=265)

# Evaluate the model on the test data.
_, mae = model.evaluate(test_dataset, verbose=0)
print(f"Test MAE: {round(mae, 3)}")


#%%
model.save('Transfomer Recommend models.h5')

# 사실 여기서 pred를 구한다음, 평가결과를 비교해야한다.

y_pred = model.predict(test_dataset)
# A, 이거 별점 예측 모델이었나.

#%%
