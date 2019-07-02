"""
Predicting the user purchase on 12.19 based on rules.

Behaviours that add_to_cart(behaviour_type = 3) and purchase(behaviour_type = 4) are closely related. Therefore, we propose a rule that Users who added the items into shopping cart on 12.18 but did not purchase will buy on the 19th.

"""
import pandas as pd

batch = 0
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
for df in pd.read_csv(open("tianchi_fresh_comp_train_user.csv", 'r'),
                      chunksize=100000):
    df_act_34 = df[df['behavior_type'].isin([3,4])]
    df_act_34.to_csv('act_34.csv',
                     columns=['time','user_id','item_id','behavior_type'],
                     index=False, header=False,
                     mode = 'a')
                     batch += 1
                     print('chunk %d done.' %batch)
            
            
data_file = open('act_34.csv', 'r')
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
df_act_34 = pd.read_csv(data_file, parse_dates = [0], date_parser = dateparse, index_col = False)
df_act_34.columns = ['time','user_id','item_id','behavior_type']
df_act_34 = df_act_34.drop_duplicates(['user_id','item_id','behavior_type']) 
#print(df_act_34.head())


df_time_3 = df_act_34[df_act_34['behavior_type'].isin(['3'])][['user_id','item_id','time']]
df_time_4 = df_act_34[df_act_34['behavior_type'].isin(['4'])][['user_id','item_id','time']]
df_time_3.columns = ['user_id','item_id', 'time3']
df_time_4.columns = ['user_id','item_id', 'time4']
df_time = pd.merge(df_time_3,df_time_4,on=['user_id','item_id'],how='outer')
df_time_34 = df_time.dropna()

#print(df_time.head())

df_time_3 = df_time[df_time['time4'].isnull()].drop(['time4'], axis = 1)
df_time_3 = df_time_3.dropna()
df_time_3.to_csv('time_3.csv', columns=['user_id','item_id','time3'], index=False)

df_time_34.to_csv('time_34.csv', columns=['user_id','item_id','time3', 'time4'], index=False)
df_time_34 = pd.read_csv("time_34.csv", parse_dates = ['time3', 'time4'], index_col = False)

delta_time = df_time_34['time4']-df_time_34['time3']
delta_hour = [] 
for i in range(len(delta_time)):
    d_hour = delta_time[i].days*24+delta_time[i]._h
    if d_hour < 0: 
        continue     # clean invalid result
    else: 
        delta_hour.append(d_hour)
        

f1 = plt.figure(1)
plt.hist(delta_hour, 30)
plt.xlabel('hours')
plt.ylabel('count')
plt.title('time decay for shopping trolley to buy 1')
plt.grid(True)
plt.show()


df_time_3 = pd.read_csv("time_3.csv", parse_dates = ['time3'], index_col = ['time3'])
ui_pred = df_time_3['2014-12-18'] 
ui_pred_in_P = pd.merge(ui_pred,P,on = ['item_id']) 

ui_pred_in_P.to_csv('tianchi_mobile_recommendation_predict.csv', columns=['user_id','item_id'],  index=False)











