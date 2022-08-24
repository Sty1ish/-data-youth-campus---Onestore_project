###############################
#####  경로, 파일, 패키지  #####
###############################
import os 
from tqdm import tqdm
import datetime
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt

# 데이터는 고객 ID별로 정리되었음.

os.chdir(r'C:\Users\9001283\Desktop\download_pts.onestorecorp.com(2022-08-02)')

value_data  = pd.read_csv('거래일시별 구매가격+id.csv', index_col = 0)

# 누적으로 구하기.
tot_arr = []
for idx in tqdm(value_data.index):
    temp = value_data.iloc[idx,]
    counter = 0
    t30arr = []
    startdate = 0
    for i, v in enumerate(temp):
        # 만약 마지막까지 올때까지 30개를 못채우면 break
        if (temp.index[i] == '유저ID'):
            break
        # 30개가 찬 순간 break
        if (len(t30arr) >= 30):
            break
        elif (pd.isna(v)==True and counter == 1):
            t30arr.append(t30arr[-1])
            continue
        elif (pd.isna(v)==False and counter == 1):
            v = v.astype(float)
            t30arr.append(t30arr[-1]+v)
            continue
        elif (pd.isna(v)==False and counter == 0):
            counter = 1
            startdate = temp.index[i]
            v = v.astype(float)
            t30arr.append(v)

            
    # 구매내역이 없는경우.
    if len(t30arr) == 0:
        t30arr = [0] * 30    
    # 만약 t30 길이 30을 만족하지 못한경우, 이후값으로 다 채워버림.
    elif len(t30arr)<30:
        t30arr.extend([t30arr[-1]] * (30-len(t30arr)))
    
    t30arr.extend([startdate , temp.loc['유저ID']])
    
    tot_arr.append(t30arr)

'''

tot_arr = []
for idx in tqdm(value_data.index):
    temp = value_data.iloc[idx,]
    counter = 0
    t30arr = []
    startdate = 0
    for i, v in enumerate(temp):
        # 만약 마지막까지 올때까지 30개를 못채우면 break
        if (temp.index[i] == '유저ID'):
            break
        # 30개가 찬 순간 break
        if (len(t30arr) >= 30):
            break
        elif (pd.isna(v)==True and counter == 1):
            t30arr.append(0)
            continue
        elif (pd.isna(v)==False and counter == 1):
            v = v.astype(float)
            t30arr.append(v)
            continue
        elif (pd.isna(v)==False and counter == 0):
            counter = 1
            startdate = temp.index[i]
            v = v.astype(float)
            t30arr.append(v)

            
    # 구매내역이 없는경우.
    if len(t30arr) == 0:
        t30arr = [0] * 30    
    # 만약 t30 길이 30을 만족하지 못한경우, 이후값으로 다 채워버림.
    elif len(t30arr)<30:
        t30arr.extend([t30arr[-1]] * (30-len(t30arr)))
    
    t30arr.extend([startdate , temp.loc['유저ID']])
    
    tot_arr.append(t30arr)

'''

t30_value_data = pd.DataFrame(np.array(tot_arr), columns=['t-30','t-29','t-28','t-27','t-26','t-25','t-24','t-23',
                                                          't-22','t-21','t-20','t-19','t-18','t-17','t-16','t-15',
                                                          't-14','t-13','t-12','t-11','t-10','t-09','t-08','t-07',
                                                          't-06','t-05','t-04','t-03','t-02','t-01','1월 이후 최초 구매일','고객ID'])

t30_value_data = t30_value_data.astype({'t-30' : 'float', 't-29' : 'float', 't-28' : 'float', 't-27' : 'float', 't-26' : 'float', 't-25' : 'float',
                                        't-24' : 'float', 't-23' : 'float', 't-22' : 'float', 't-21' : 'float', 't-20' : 'float', 't-19' : 'float',
                                        't-18' : 'float', 't-17' : 'float', 't-16' : 'float', 't-15' : 'float', 't-14' : 'float', 't-13' : 'float',
                                        't-12' : 'float', 't-11' : 'float', 't-10' : 'float', 't-09' : 'float', 't-08' : 'float', 't-07' : 'float',
                                        't-06' : 'float', 't-05' : 'float', 't-04' : 'float', 't-03' : 'float', 't-02' : 'float', 't-01' : 'float'})
t30_value_data.info()

t30_value_data.head()



plot_data = t30_value_data.drop(['1월 이후 최초 구매일', '고객ID'],axis = 1)
plot_data = plot_data.T









# 아래는 플롯 데이터.







plt.figure(figsize=(44,22))
col_list = plot_data.columns

for col in tqdm(col_list):
    y = plot_data[col]
    plt.plot(y)
    #if (col in [10000,20000,30000,40000,50000,60000,70000,82365]):
    #    plt.show()
    #    plt.savefig(str(col)+'번째까지 최초구매일 누적 구매액 패턴.png')
    #    plt.figure(figsize=(44,22))

# if문 비활성화 하고 이정도.
plt.show()
plt.savefig('최초구매일 일간 구매액 패턴.png')  



# amount_data = pd.read_csv('거래일시별 구매횟수+id.csv')

# 연속으로 못그림. 위에 플롯 지우고 새로 실행해야 램 부족 이슈 안발생
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
plot_data_scaled = pd.DataFrame(scaler.fit_transform(plot_data))


plt.figure(figsize=(41,16))
col_list = plot_data_scaled.columns

for col in tqdm(col_list):
    y = plot_data_scaled[col]
    plt.plot(y)

plt.show()
plt.savefig('최초구매일 일간 구매액 패턴-minmax+scaled.png')


# 연속으로 못그림. 위에 플롯 지우고 새로 실행해야 램 부족 이슈 안발생
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
plot_data_scaled = pd.DataFrame(scaler.fit_transform(plot_data.T)).T


plt.figure(figsize=(41,16))
col_list = plot_data_scaled.columns

for col in tqdm(col_list):
    y = plot_data_scaled[col]
    plt.plot(y)

plt.show()
plt.savefig('최초구매일 일간 구매액 패턴(시점기준-minmax+scaled).png')




# 상한이랑 하한을 정해놓고, minmax scaleing -> 1초과 1미만 나오라는 뜻.

lowlim  = -100000
highlim =  100000
custom_scaled_plot_data = plot_data.apply(lambda x : (x - lowlim) / (highlim - lowlim))


plt.figure(figsize=(41,16))
col_list = custom_scaled_plot_data.columns

for col in tqdm(col_list):
    y = custom_scaled_plot_data[col]
    plt.plot(y)

plt.show()
plt.savefig('최초구매일 누적 구매액 패턴(custom-minmax+scaled).png')
