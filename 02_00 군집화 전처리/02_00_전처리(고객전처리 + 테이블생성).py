#!/usr/bin/env python
# coding: utf-8

# ### 데이터 로드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import notebook
import os
os.chdir('P:\새 폴더')
get_ipython().run_line_magic('matplotlib', 'inline')


df_purchase = pd.read_csv("원스토어_구매내역.csv", names=['구매일자', '구매시간대', '사용자ID', '성별','나이','구매자_통신사','상품ID','대표상품ID','상품카테고리','상품세부카테고리','상품등급','구매가격'])

# 상품 ID, ~ 카테고리 수치형으로
df_purchase['상품ID'] = df_purchase['상품ID'].str[1:]
df_purchase['대표상품ID'] = df_purchase['대표상품ID'].str[1:]
df_purchase['상품카테고리'] = df_purchase['상품카테고리'].str[2:]
df_purchase['상품세부카테고리'] = df_purchase['상품세부카테고리'].str[2:]

# 요일 추출
df_purchase['구매일자'] = pd.to_datetime(df_purchase['구매일자'].astype('str'), format='%Y/%m/%d')

import datetime
df_purchase.insert(1, '구매요일', df_purchase['구매일자'].dt.weekday) # (0-월, 1-화, 2-수, ...)

df_purchase['구매일자'] = df_purchase['구매일자'].apply(lambda x : x.strftime('%Y%m%d'))

df_purchase[['상품ID','대표상품ID','상품카테고리','상품세부카테고리']] = df_purchase[['상품ID','대표상품ID','상품카테고리','상품세부카테고리']].astype('int32')

df_purchase['성별'] = df_purchase['성별'].replace('M', 0).replace('F', 1).replace('Z', np.nan)
df_purchase['나이'].replace('ZZZ', 999, inplace=True)
df_purchase['나이'] = df_purchase['나이'].astype('int32')
df_purchase['구매자_통신사'] = df_purchase['구매자_통신사'].replace('SKT', 0).replace('U+', 1).replace('KT', 2)
df_purchase['상품등급'] = df_purchase['상품등급'].replace('전체이용가', 0).replace('청소년이용불가', 1)


df_purchase['구매일자'] = df_purchase['구매일자'].astype('str').str[4:]

df_purchase.to_csv("./purchase.csv", encoding='utf-8-sig')


#%%


# csv 파일로 저장한 purchase 파일 불러옴
df_purchase = pd.read_csv('purchase.csv', index_col=0)

df_click = pd.read_csv("산학_클릭로그.csv", names=['클릭일자', '클릭시간대', '사용자ID', '대표상품ID'])

df_click['대표상품ID'] = df_click['대표상품ID'].str[1:]
df_click['대표상품ID'] = df_click['대표상품ID'].astype('int32')
df_click['클릭일자'] = df_click['클릭일자'].astype('int32')
df_click['클릭시간대'] = df_click['클릭시간대'].astype('int32')

df_purchase_march = df_purchase[df_purchase['구매일자'].astype('str').str[:1] == '3']

df_purchase_march1 = df_purchase_march[df_purchase_march.구매가격 >= 0]

df_purchase_march2 = df_purchase_march1.groupby(['구매일자', '구매요일', '구매시간대', '사용자ID', '성별', '나이', '구매자_통신사', '상품ID', '대표상품ID', '상품카테고리', '상품세부카테고리', '상품등급']).sum().reset_index()

df_purchase_march2.to_csv("./02_00고객데이터사전작업.csv", encoding='utf-8-sig')


#%%


# 구매내역 테이블 생성
df_purchase = pd.read_csv('02_00고객데이터사전작업.csv', index_col=0)

# 클릭로그 테이블 생성
df_click = pd.read_csv("산학_클릭로그.csv", names=['클릭일자', '클릭시간대', '사용자ID', '대표상품ID'])
df_click['대표상품ID'] = df_click['대표상품ID'].str[1:]
df_click['대표상품ID'] = df_click['대표상품ID'].astype('int32')
df_click['클릭일자'] = df_click['클릭일자'].astype('int32')
df_click['클릭시간대'] = df_click['클릭시간대'].astype('int32')


# ### customer table 생성

# 클릭로그 테이블에서 총 사용자 명수 추출
userID = df_click.사용자ID.unique()

# 사용자별 성별 추출(최빈값)
a = df_purchase[df_purchase.사용자ID.isin(userID)].groupby('사용자ID')['성별'].agg(**{
    '성별':lambda x:x.mode()[0]
}).reset_index()

# 사용자별 나이 추출(최빈값)
b = df_purchase[df_purchase.사용자ID.isin(userID)].groupby('사용자ID')['나이'].agg(**{
    '나이':lambda x:x.mode()[0]
}).reset_index()

# 사용자별 구매자_통신사 추출(최빈값)
c = df_purchase[df_purchase.사용자ID.isin(userID)].groupby('사용자ID')['구매자_통신사'].agg(**{
    '구매자_통신사':lambda x:x.mode()[0]
}).reset_index()

# a, b, c 병합
customer_table = a.merge(b).merge(c).set_index('사용자ID').reset_index()



# ### 구매성향(비율) 데이터

# df_purchase 복사
df_purchase_copy = df_purchase.copy()

df_purchase_copy.상품세부카테고리 = 'buy_' + df_purchase_copy.상품세부카테고리.astype(str)

# 중첩 dictionary를 사용하여 사용자ID별 구매비율 추출
purchase_rate = {}
for idx, value in notebook.tqdm(df_purchase_copy.groupby('사용자ID')):
    purchase_rate[idx] = dict(value.상품세부카테고리.value_counts() / len(value.상품세부카테고리))

# 중첩 dictionary를 dataframe으로 변환
output_purchase_rate = pd.DataFrame(purchase_rate).transpose()

# 사용자ID index를 column으로 빼내기
output_purchase_rate = output_purchase_rate.reset_index().rename(columns={"index": "사용자ID"})

result_purchase_rate = output_purchase_rate.fillna(0)

result_purchase_rate.to_csv("./02_01고객구매성향.csv", encoding='utf-8-sig')


# ###  클릭시간대
# - **대중교통 증회운행시간 및 교통체증 시간대 기반으로 시간대 분류**
# - 통근시간 : 오전 6시 ~ 오전 10시 & 오후 5시 ~ 9시
# - 주간시간 : 오전 10시 ~ 오후 5시
# - 심야시간 : 오후 9시 ~ 오전 6시
# - 통근시간 : 0, 주간시간 : 1, 심야시간 : 2

# click data copy
df_click_copy = df_click.copy()

# 시간대 설정. 통근시간 : 0, 주간시간 : 1, 심야시간 : 2
time_setting = {}
for i in range(24):
    if ((i >= 6) & (i <= 9)) | ((i >= 17) & (i <= 20)):
        time_setting[i] = 0
    elif (i >= 10) & (i <= 16):
        time_setting[i] = 1
    else:
        time_setting[i] = 2

# 시간대 치환 / 통근시간 : 0, 주간시간 : 1, 심야시간 : 2
df_click_copy['클릭시간대'] = df_click_copy['클릭시간대'].map(time_setting)

# 중첩 dictionary를 사용하여 사용자ID별 클릭시간 추출
click_time = {}
for idx, value in notebook.tqdm(df_click_copy.groupby('사용자ID')):
    click_time[idx] = dict(value.클릭시간대.value_counts() / len(value.클릭시간대))
    
# 중첩 dictionary를 dataframe으로 변환
output_click_time = pd.DataFrame(click_time).transpose()

# 사용자ID index를 column으로 빼내기
result_click_time = output_click_time.reset_index().rename(columns={"index": "사용자ID", 2 : '심야시간', 1 : '주간시간', 0 : '통근시간'})

result_click_time = result_click_time.fillna(0)

result_click_time.to_csv("./02_01고객클릭시간대.csv", encoding='utf-8-sig')



# ### 고객구매전환율
# - 고유 구매 대표상품ID / 고유 클릭 대표상품ID

# 사용자별 고유 구매 대표상품ID 개수 딕셔너리화
purchase_dict = {}
for idx, value in notebook.tqdm(df_purchase.groupby('사용자ID')):
    purchase_dict[idx] = len(value.대표상품ID.unique())

# 사용자별 고유 클릭 대표상품ID 개수 딕셔너리화
click_dict = {}
for idx, value in notebook.tqdm(df_click.groupby('사용자ID')):
    click_dict[idx] = len(value.대표상품ID.unique())

purchase_click_dict = {}
for key, value in notebook.tqdm(purchase_dict.items()):
    try:
        purchase_click_dict[key] = value / click_dict[key]
    except:
        purchase_click_dict[key] = np.nan

result_CVR = pd.DataFrame(pd.Series(purchase_click_dict)).reset_index().rename(columns={"index": "사용자ID", 0 : '구매전환율'})

result_CVR.to_csv("./02_01구매전환율.csv", encoding='utf-8-sig')


# ###  클릭비율(성향) 데이터

df_click_copy = df_click.copy()

df_click_dtl = pd.read_csv('산학_클릭로그_상품카테고리.csv', names=['대표상품ID', '상품카테고리', '분류', '상품세부카테고리', '카테고리명'])

df_click_dtl['대표상품ID'] = df_click_dtl['대표상품ID'].str[1:]
df_click_dtl['대표상품ID'] = df_click_dtl['대표상품ID'].astype('int32')
df_click_dtl['상품세부카테고리'] = df_click_dtl['상품세부카테고리'].str[2:]
df_click_dtl['상품세부카테고리'] = df_click_dtl['상품세부카테고리'].astype('int32')

df_click_dtl = df_click_dtl[['대표상품ID', '상품세부카테고리', '카테고리명']]

kkey = np.array(df_click_dtl['대표상품ID'])
vvalues = np.array(df_click_dtl['상품세부카테고리'])

key_values = {}
for i, j in zip(kkey, vvalues):
    key_values[i] = j

df_click_copy['상품세부카테고리'] = df_click_copy.대표상품ID.map(key_values)

df_click_copy.상품세부카테고리 = 'click_' + df_click_copy.상품세부카테고리.astype(str)

# 사용자ID 별 클릭비율 딕셔너리화
click_rate = {}
for idx, value in notebook.tqdm(df_click_copy.groupby('사용자ID')):
    click_rate[idx] = dict(value.상품세부카테고리.value_counts()/len(value.상품세부카테고리))

result_click_rate = pd.DataFrame(click_rate).transpose()

# 사용자ID index를 column으로 빼내기
result_click_rate = result_click_rate.reset_index().rename(columns={"index": "사용자ID"})

result_click_rate = result_click_rate.fillna(0)

result_click_rate.to_csv("./02_01고객클릭성향.csv", encoding='utf-8-sig')



# ### 구매금액 데이터
result_purchase_amount = pd.DataFrame(df_purchase.groupby('사용자ID').구매가격.sum().reset_index())



# ### 데이터 합치기
# 최종 테이블 생성
final_result = customer_table.merge(result_purchase_rate, on='사용자ID').merge(result_click_rate, on='사용자ID').merge(result_click_time, on='사용자ID').merge(result_CVR, on='사용자ID').merge(result_purchase_amount, on='사용자ID')

final_result.to_csv("./02_01최종구매내역테이블.csv", encoding='utf-8-sig')