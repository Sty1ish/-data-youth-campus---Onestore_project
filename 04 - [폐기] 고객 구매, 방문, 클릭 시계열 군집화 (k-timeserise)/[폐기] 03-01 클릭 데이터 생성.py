# 거래일시별 구매가격+id.csv
# 거래일시별 구매횟수+id.csv
# 이 두개는 그대로 들고가고.
# 클릭 횟수를 측정한다.

###############################
#####  경로, 파일, 패키지  #####
###############################
import os 
from tqdm import tqdm
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


os.chdir(r'C:\Users\9001283\Desktop\download_pts.onestorecorp.com(2022-08-02)')

data = pd.read_csv('산학_클릭로그.csv', header = None, names = ['거래일시','거래시간','유저ID','대표상품ID'])

# 알고리즘 효율을 위해, 시간 순서대로 정렬한다.
data = data.sort_values(by=['거래일시', '거래시간'] ,ascending=True).reset_index(drop=True)
data.head() # 정렬되었다.

# 3월 한달간 고유 유저의 수.
print(f'고유 유저수 : {len(data.유저ID.unique())}')

# 이 데이터셋을 첫 클릭일자 기준으로 보는게 맞을까? 아니면 그냥 전체를 보는게 맞을까.
# 전체의 개수를 본다면 이건 교차표로 보는게 맞는 데이터셋이다.

cross_table = pd.crosstab(data['거래일시'], data['유저ID'])

# 그걸 그래프로 투명도 주고 그리면 다음과 같지.

plt.figure(figsize=(66,44))
col_list = cross_table.columns

for col in tqdm(col_list):
    line = cross_table[col]
    sns.lineplot(data = line, alpha = 0.2)

# if문 비활성화 하고 이정도.
plt.xlabel('Day30')
plt.ylabel('user-id')
plt.show()
plt.savefig('클릭 데이터 패턴분석 .png')  




'''
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

'''
    
















# 월 일 데이터 만들고, 요일 정보 반환.
#data = data.astype({'거래일시' : 'str'})
#data['월'] = data.거래일시.apply(lambda x : x[4:6]).astype('int')
#data['일'] = data.거래일시.apply(lambda x : x[6:8]).astype('int')
#data.거래일시 = pd.to_datetime(data.거래일시)
#data['구매요일'] = data.거래일시.dt.weekday # 0이 월요일, 6이 일요일.

# tempdf.거래일시.iloc[0] Timestamp('2020-01-05 00:00:00') 첫 날짜.
# tempdf.거래일시.iloc[7] + datetime.timedelta(days = 1) 이렇게 하루를 더할수 있다.
# 날짜의 끝은 2020-03-31 Timestamp('2020-03-31 00:00:00')
# 이 연산으로 참인지 거짓인지 알수있다. datetime.datetime(2020, 3, 31) == data.거래일시.iloc[-1]
    
