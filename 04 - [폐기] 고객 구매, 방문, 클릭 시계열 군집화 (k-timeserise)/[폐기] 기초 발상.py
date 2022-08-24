###############################
#####  경로, 파일, 패키지  #####
###############################
import os 
from tqdm import tqdm
import datetime
import pandas as pd
import numpy as np
import pandas_profiling


os.chdir(r'C:\Users\9001283\Desktop\download_pts.onestorecorp.com(2022-08-02)')

data = pd.read_csv('원스토어_구매내역.csv', header = None,
                   names = ['거래일시','거래시간','유저ID','성별','나이',
                            '통신사','상품ID','대표상품ID','카테고리',
                            '세부카테고리','상품등급','구매가격'])

# 귀찮은 초반 EDA
# pr = data.profile_report()
# pr.to_file('./pr_report.html')

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

# 월 일 데이터 만들고, 요일 정보 반환.
data = data.astype({'거래일시' : 'str'})
data['월'] = data.거래일시.apply(lambda x : x[4:6]).astype('int')
data['일'] = data.거래일시.apply(lambda x : x[6:8]).astype('int')
data.거래일시 = pd.to_datetime(data.거래일시)
data['구매요일'] = data.거래일시.dt.weekday # 0이 월요일, 6이 일요일.

data.info() # 최종 사용 메모리. 533.8mb

# 알고리즘 효율을 위해, 시간 순서대로 정렬한다.
data = data.sort_values(by=['거래일시', '거래시간'] ,ascending=True).reset_index(drop=True)
data.head() # 정렬되었다.

###############################
##### 데이터 셋 기본 세팅. #####
###############################

# 시간흐름에 따른 거래일시와, 거래시간별 군집화

# 3달간 유저수.
print(f'고유 유저수 : {len(data.유저ID.unique())}')


#################################################
##### 일자당 구매액, 방문액수 데이터 셋 제작  #####
#################################################
# 42~48시간 소요? 아.... 장난없네 이거. 효율성 문제였다. 1~2시간컷 코드로 변경.
# tempdf를 매번 만들고 검색하는 과정이 이미 비효율적인데 이거 할수 있는방법 없나.
# 만들고 싶은 프레임.
# 유저 ID, 첫거래일, D+1 ~ D+30일 까지의 합계구매액과, 구매 횟수. (60개 열인데, 개인적으로 두개 군집을 다르게 보고싶음. 변수가 다르니까.)

dates = [datetime.datetime(2020,1,1) + datetime.timedelta(days = i) for i in range(31+28+31)]
null_frame = pd.DataFrame({'거래일시' : dates})

# 첫번째는 그냥 실행.
user_dict = data.유저ID.unique()
user0 = user_dict[0]

tempdf = data[data.유저ID == user0]
temp_val = tempdf.groupby('거래일시').agg({'구매가격' : 'sum'})
temp_size = tempdf.groupby('거래일시').size().to_frame(name = '구매횟수')
temp_val = pd.merge(null_frame, temp_val, how='left', on='거래일시').구매가격.to_numpy()
temp_size = pd.merge(null_frame, temp_size, how='left', on='거래일시').구매횟수.to_numpy()

arr_val = np.array([temp_val])
arr_size = np.array([temp_size])
user_id = [user0]



for user, tempdf in tqdm(data[data.유저ID != user0].groupby('유저ID')):
    temp_val = tempdf.groupby('거래일시').agg({'구매가격' : 'sum'})
    temp_size = tempdf.groupby('거래일시').size().to_frame(name = '구매횟수') 
    temp_val = pd.merge(null_frame, temp_val, how='left', on='거래일시').구매가격.to_numpy()
    temp_size = pd.merge(null_frame, temp_size, how='left', on='거래일시').구매횟수.to_numpy()
    
    arr_val = np.append(arr_val,[temp_val],axis= 0)
    arr_size = np.append(arr_size,[temp_size],axis= 0)
    user_id.append(user)
    
arr_val = pd.DataFrame(arr_val, columns = dates)
arr_size = pd.DataFrame(arr_size, columns = dates)

# 판별 기준값 유저 ID 추가.
arr_val['유저ID'] = user_id
arr_size['유저ID'] = user_id

arr_val.to_csv('거래일시별 구매가격+id.csv')
arr_size.to_csv('거래일시별 구매횟수+id.csv')

del dates, null_frame, user_dict, user0, user, tempdf, temp_val, temp_size


    
    # tempdf.거래일시.iloc[0] Timestamp('2020-01-05 00:00:00') 첫 날짜.
    # tempdf.거래일시.iloc[7] + datetime.timedelta(days = 1) 이렇게 하루를 더할수 있다.
    # 날짜의 끝은 2020-03-31 Timestamp('2020-03-31 00:00:00')
    # 이 연산으로 참인지 거짓인지 알수있다. datetime.datetime(2020, 3, 31) == data.거래일시.iloc[-1]
    
    
# 이제 이 파일에 대한 분석은 다른 파일에서 진행함. (거래일시별 구매가격, 횟수 군집화.)




###############################
##### 데이터 셋 기본 세팅. #####
###############################

# 고객별 카테고리 소비성향별 군집화.