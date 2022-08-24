###############################
#####  경로, 파일, 패키지  #####
###############################
import os 
from tqdm import tqdm
import datetime
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터는 고객 ID별로 정리되었음.

os.chdir(r'C:\Users\9001283\Desktop\download_pts.onestorecorp.com(2022-08-02)')

data = pd.read_csv('거래일시별 구매가격+id.csv', index_col = 0)

########## 단순 반복작업 함수화 ############

def make_t30df(value_data, continuous):
    tot_arr = []
    if (continuous == 0): # 최초 구매일 기준 30일간 구매내역 구하기.
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
                t30arr.extend([0] * (30-len(t30arr)))
            
            t30arr.extend([startdate , temp.loc['유저ID']])
            
            tot_arr.append(t30arr)
    else: # 최초구매일 기준 30일치 누적합 구하기.
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
            
    t30_value_data = pd.DataFrame(np.array(tot_arr), columns=['t-30','t-29','t-28','t-27','t-26','t-25','t-24','t-23',
                                                              't-22','t-21','t-20','t-19','t-18','t-17','t-16','t-15',
                                                              't-14','t-13','t-12','t-11','t-10','t-09','t-08','t-07',
                                                              't-06','t-05','t-04','t-03','t-02','t-01','1월 이후 최초 구매일','고객ID'])

    t30_value_data = t30_value_data.astype({'t-30' : 'float', 't-29' : 'float', 't-28' : 'float', 't-27' : 'float', 't-26' : 'float', 't-25' : 'float',
                                            't-24' : 'float', 't-23' : 'float', 't-22' : 'float', 't-21' : 'float', 't-20' : 'float', 't-19' : 'float',
                                            't-18' : 'float', 't-17' : 'float', 't-16' : 'float', 't-15' : 'float', 't-14' : 'float', 't-13' : 'float',
                                            't-12' : 'float', 't-11' : 'float', 't-10' : 'float', 't-09' : 'float', 't-08' : 'float', 't-07' : 'float',
                                            't-06' : 'float', 't-05' : 'float', 't-04' : 'float', 't-03' : 'float', 't-02' : 'float', 't-01' : 'float'})
    print(t30_value_data.info())
    print(f'데이터 프레임 형태 (t30_value_data.shape)')
    print(t30_value_data.head())
    
    return t30_value_data

############ 단순 코드 반복인데 효율성 위해서 코드 짜면
def userline_plot_sns(table, x_lab, y_lab, plot_title):
    plt.figure(figsize=(22,11))
    matplotlib.rc('font',family='gulim')
    col_list = table.columns

    for col in tqdm(col_list):
        line = table[col]
        sns.lineplot(data = line, alpha = 0.2)

    plt.xlabel(str(x_lab))
    plt.ylabel(str(y_lab))
    plt.title(str(plot_title), fontsize=25)
    plt.show()
    plt.savefig(str(plot_title)+'(alpha = 0.2).png')

def userline_plot_pyplot(table, x_lab, y_lab, plot_title):
    plt.figure(figsize=(22,11))
    matplotlib.rc('font',family='gulim')
    col_list = table.columns

    for col in tqdm(col_list):
        y = table[col]
        plt.plot(y)

    plt.xlabel(str(x_lab))
    plt.ylabel(str(y_lab))
    plt.title(str(plot_title), fontsize=25)
    plt.show()
    plt.savefig(str(plot_title)+'(pyplot).png')

###########################################
###########################################
###########################################
# 소비 금액 그래프
###########################################
###########################################
###########################################
# 첫 구매일 기준, 일간 소비 패턴 그래프. (투명도 포함 - 30분-1시간 소요.)
# 아래 블럭 한번에 실행.
plot_data = make_t30df(data, continuous = 0).drop(['1월 이후 최초 구매일', '고객ID'],axis = 1)
plot_data = plot_data.T


# pyplot
userline_plot_pyplot(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 일간 소비 금액' )
# sns
# userline_plot_sns(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 일간 소비 금액' )

# 정규화 시켜봐야 결국 형태는 안변했음.





###########################################
###########################################
###########################################
# 첫 구매일 기준, 30일간 누적 소비 금액 그래프. (투명도 포함 - 30분-1시간 소요.)
# 아래 블럭 한번에 실행.
plot_data = make_t30df(data, continuous = 1).drop(['1월 이후 최초 구매일', '고객ID'],axis = 1)
plot_data = plot_data.T


# pyplot
userline_plot_pyplot(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 한달 누적 소비 금액' )
# sns
# userline_plot_sns(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 한달 누적 소비 금액' )

###########################################
###########################################
###########################################
# 그냥 91일간 전체 흐름을 볼까? 누적은 보지말고. 전체 기간 그래프를.

plot_data = data.drop(['유저ID'],axis = 1).fillna(0).T


# pyplot
userline_plot_pyplot(plot_data, x_lab = 'Day1~Day91', y_lab = 'user-id per line', plot_title = '1-3월 일간 소비 금액' )
# sns
# userline_plot_sns(plot_data, x_lab = 'Day1~Day91', y_lab = 'user-id per line', plot_title = '1-3월 일간 소비 금액' )












###########################################
###########################################
###########################################
#### 구매 횟수 기준으로 필터링. ####
###########################################
###########################################
###########################################

data = pd.read_csv('거래일시별 구매횟수+id.csv', index_col = 0)

###########################################
###########################################
###########################################
# 첫 구매일 기준, 일간 구매 횟수 그래프. (투명도 포함 - 30분-1시간 소요.)
# 아래 블럭 한번에 실행.
plot_data = make_t30df(data, continuous = 0).drop(['1월 이후 최초 구매일', '고객ID'],axis = 1)
plot_data = plot_data.T




# pyplot
userline_plot_pyplot(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 일간 구매횟수' )
# sns
# userline_plot_sns(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 일간 구매횟수' )





###########################################
###########################################
###########################################
# 첫 구매일 기준, 30일간 누적 구매 횟수 그래프. (투명도 포함 - 30분-1시간 소요.)
# 아래 블럭 한번에 실행.
plot_data = make_t30df(data, continuous = 1).drop(['1월 이후 최초 구매일', '고객ID'],axis = 1)
plot_data = plot_data.T


# pyplot
userline_plot_pyplot(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 30일 누적 구매횟수' )
# sns
# userline_plot_sns(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '첫 구매일 기준 30일 누적 구매횟수' )




###########################################
###########################################
###########################################
# 그냥 91일간 전체 흐름을 볼까? 누적은 보지말고. 전체 기간 그래프를.

plot_data = data.drop(['유저ID'],axis = 1).fillna(0).T



# pyplot
userline_plot_pyplot(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '1-3월 일간 구매횟수' )
# sns
# userline_plot_sns(plot_data, x_lab = 't-30~t-1', y_lab = 'user-id per line', plot_title = '1-3월 일간 구매횟수' )




del data, plot_data


###########################################
###########################################
###########################################
#### 클릭 데이터 그래프. ####
###########################################
###########################################
###########################################
click_data = pd.read_csv('산학_클릭로그.csv', header = None, names = ['거래일시','거래시간','유저ID','대표상품ID'])

# 알고리즘 효율을 위해, 시간 순서대로 정렬한다.
click_data = click_data.sort_values(by=['거래일시', '거래시간'] ,ascending=True).reset_index(drop=True)


# 전체의 개수를 본다면 이건 교차표로 보는게 맞는 데이터셋이다.

cross_table = pd.crosstab(click_data['거래일시'], click_data['유저ID'])

# 그걸 그래프로 투명도 주고 그리면 다음과 같지.


# pyplot
userline_plot_pyplot(cross_table, x_lab = 'March 1~30', y_lab = 'user-id per line', plot_title = '3월 일간 클릭 횟수 그래프' )
# sns
# userline_plot_sns(cross_table, x_lab = 'March 1~30', y_lab = 'user-id per line', plot_title = '3월 일간 클릭 횟수 그래프' )
