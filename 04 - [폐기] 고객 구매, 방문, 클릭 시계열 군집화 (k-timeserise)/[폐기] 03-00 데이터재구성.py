# 기본 발상. 구매액이 다른데,
# 나머지 앞열이 모두 같다면, 데이터는 같은게 아닐까?
# 음수는 환불을 의미할 것인데, 그러면 하나의 구매에 두개의 데이터가 잡히지 않을까?
# 따라서 구매횟수를 계산하기 위한 데이터셋을 재구성하였음.
# 환불액수는 반영되는것이 올바르다 판단 03-00 기초발상의 '거래일시별 구매가격+id.csv'을 그대로 사용함.

import os 
from tqdm import tqdm
import datetime
import pandas as pd
import numpy as np


os.chdir(r'C:\Users\9001283\Desktop\download_pts.onestorecorp.com(2022-08-02)')

data = pd.read_csv('원스토어_구매내역.csv', header = None,
                   names = ['거래일시','거래시간','유저ID','성별','나이',
                            '통신사','상품ID','대표상품ID','카테고리',
                            '세부카테고리','상품등급','구매가격'])

###############################
##### 데이터 셋 기본 세팅. #####
###############################

# 데이터셋 작업전처리
data.info() # 일단 895mb 이상 메모리를 사용하고 있음.

data.나이.unique() # 데이터셋 그대로 ZZZ가 결측값
data.loc[data.나이 == 'ZZZ', '나이'] = 999     # 결측값 999로 대체함.

# 데이터셋 메모리 절약을 위해 categori화.
data = data.astype({'성별' : 'category', '나이': 'int', '통신사' : 'category', '상품등급' : 'category', '거래시간' : 'category',
                    '상품ID' : 'category', '대표상품ID' : 'category', '카테고리' : 'category', '세부카테고리' : 'category'})

data.info() # 메모리 384mb 이상으로 사용중. 반정도 아낌.

# 알고리즘 효율을 위해, 시간 순서대로 정렬한다.
data = data.sort_values(by=['거래일시', '거래시간'] ,ascending=True).reset_index(drop=True)
data.head() # 정렬되었다
# 977,5913개의 데이터에서 977,5191개 데이터로 감소..

# 메모리 부족 이슈로 추가로 바꿔줌.
data = data.astype({'월' : 'category', '나이': 'category', '유저ID': 'category'})

# data = data[data.구매가격 >= 0]
# 977,5913개의 데이터에서 977,5191개 데이터로 감소.

# 3개월 동시에 진행불가로, 이렇게 진행함.
# ['DP14', 'DP26', 'DP29', 'DP31']
data1 = data[data.거래일시.str[5:6] == '1']
data1.info()
data1 = data1.astype({'거래일시' : 'category'})
data1 = data1.groupby(['거래일시','거래시간', '유저ID', '성별', '나이', '통신사', '상품ID', '대표상품ID', '카테고리', '세부카테고리', '상품등급']).sum().reset_index()





    
# 월 # 일 데이터 만들고, 요일 정보 반환.
# data = data.astype({'거래일시' : 'str'})
# data['월'] = data.거래일시.apply(lambda x : x[4:6]).astype('int')
# data['일'] = data.거래일시.apply(lambda x : x[6:8]).astype('int')
# data.거래일시 = pd.to_datetime(data.거래일시)
# data['구매요일'] = data.거래일시.dt.weekday # 0이 월요일, 6이 일요일.
    
data.to_csv("./02_00고객데이터사전작업.csv", encoding='utf-8-sig')