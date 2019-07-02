""""
Constructing the following feature:

******user********
u_b_count_in_n（n=1/3/6): the total number of behaviors of users n days before the test day (feature = 3)
u_bi_count_in_n（i=1/2/3/4,n=1/3/6): User counts of each behavior n days before the test day (feature = 12)
u_b4_rate: User click-through conversion rate (feature =1)
u_b4_diff_hours: User click-to-buy average time difference (feature =1)

******item********
i_u_count_in_n(n=1/3/6): the total number of users on the item n days before the test date (feature = 3)
i_b_count_in_n(n=1/3/6): the total number of behaviors of users on the item n days before the test day (feature = 3)
i_bi_count_in_n (i=1/2/3/4,n=1/3/6): User counts of each behavior on the item n days before the test day (feature = 12)
i_b4_rate: item click-through conversion rate (feature =1)
i_b4_diff_hours: item click-to-buy average time difference (feature =1)

******category********
c_u_count_in_n(n=1/3/6): the total number of users on the category n days before the test date (feature = 3)
c_b_count_in_n(n=1/3/6): the total number of behaviors of users on the category n days before the test day (feature = 3)
c_bi_count_in_n(i=1/2/3/4,n=1/3/6): User counts of each behavior on the category n days before the test day (feature = 12)
c_b4_rate: category click-through conversion rate (feature =1)
c_b4_diff_hours: category click-to-buy average time difference (feature =1)

*****item-category******
ic_u_rank_in_c: Sort the item based on the number of users in the category (feature =1)
ic_b_rank_in_c: Sort the item based on the number of user behaviours in the category (feature =1)
ic_b4_rank_in_c: Sort the item based on the number of user purchase behaviours in the category (feature =1)

*****item-user******
ui_b_count_in_n: the total number of behaviors for item-user pairs n days before the test day (feature =3)
ui_bi_count_in_n: the total number of each behavior for item-user pairs n days before the test day (feature =12)
ui_bi_last_hours: each behaviour time difference between the last occurrence and the test day for item-user(feature =4)
ui_b_count_rank_in_n_in_u: sort the behavior of the user-item pair in all of the user's items (feature =3)
ui_b_count_rank_in_n_in_uc: sort the behavior of the user-item pair in user-category pair (feature =3)

*****user - category******
uc_b_count_in_n:  the total number of behaviors for user-category pairs n days before the test day (feature =3)
uc_bi_count_in_n: the total number of each behavior for user-category pairs n days before the test day (feature =12)
uc_bi_last_hours: each behaviour time difference between the last occurrence and the test day for user-category (feature =4)
uc_b_count_rank_in_n_in_u: sort the behavior of the user-category pair in all of the user's items (feature =3)

""""

##### file path
# input 
path_df_D = "tianchi_fresh_comp_train_user.csv"

path_df_part_1 = "mobile/df_part_1.csv"
path_df_part_2 = "mobile/df_part_2.csv"
path_df_part_3 = "mobile/df_part_3.csv"

path_df_part_1_tar = "mobile/df_part_1_tar.csv"
path_df_part_2_tar = "mobile/df_part_2_tar.csv"

path_df_part_1_uic_label = "mobile/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "mobile/df_part_2_uic_label.csv"
path_df_part_3_uic       = "mobile/df_part_3_uic.csv"

# output
path_df_part_1_U   = "mobile/feature/df_part_1_U.csv"  
path_df_part_1_I   = "mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "mobile/feature/df_part_1_UC.csv"

path_df_part_2_U   = "mobile/feature/df_part_2_U.csv"  
path_df_part_2_I   = "mobile/feature/df_part_2_I.csv"
path_df_part_2_C   = "mobile/feature/df_part_2_C.csv"
path_df_part_2_IC  = "mobile/feature/df_part_2_IC.csv"
path_df_part_2_UI  = "mobile/feature/df_part_2_UI.csv"
path_df_part_2_UC  = "mobile/feature/df_part_2_UC.csv"

path_df_part_3_U   = "mobile/feature/df_part_3_U.csv"  
path_df_part_3_I   = "mobile/feature/df_part_3_I.csv"
path_df_part_3_C   = "mobile/feature/df_part_3_C.csv"
path_df_part_3_IC  = "mobile/feature/df_part_3_IC.csv"
path_df_part_3_UI  = "mobile/feature/df_part_3_UI.csv"
path_df_part_3_UC  = "mobile/feature/df_part_3_UC.csv"


import pandas as pd
import numpy as np

path_df = open(path_df_part_1, 'r')
df_part_1 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
df_part_1.columns = ['time','user_id','item_id','behavior_type','item_category']

df_part_1['cumcount'] = df_part_1.groupby(['user_id', 'behavior_type']).cumcount()
df_part_1_u_b_count_in_6 = df_part_1.drop_duplicates(['user_id','behavior_type'], 'last')[['user_id','behavior_type','cumcount']]

print(df_part_1.head())
print(df_part_1_u_b_count_in_6.head())


df_part_1_u_b_count_in_6 = pd.get_dummies(df_part_1_u_b_count_in_6['behavior_type']).join(df_part_1_u_b_count_in_6[['user_id','cumcount']])
df_part_1_u_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_u_b_count_in_6.head()


df_part_1_u_b_count_in_6['u_b1_count_in_6'] = df_part_1_u_b_count_in_6['behavior_type_1'] * (df_part_1_u_b_count_in_6['cumcount']+1)
df_part_1_u_b_count_in_6['u_b2_count_in_6'] = df_part_1_u_b_count_in_6['behavior_type_2'] * (df_part_1_u_b_count_in_6['cumcount']+1)
df_part_1_u_b_count_in_6['u_b3_count_in_6'] = df_part_1_u_b_count_in_6['behavior_type_3'] * (df_part_1_u_b_count_in_6['cumcount']+1)
df_part_1_u_b_count_in_6['u_b4_count_in_6'] = df_part_1_u_b_count_in_6['behavior_type_4'] * (df_part_1_u_b_count_in_6['cumcount']+1)
df_part_1_u_b_count_in_6.head()



df_part_1_u_b_count_in_6 = df_part_1_u_b_count_in_6.groupby('user_id').agg({'u_b1_count_in_6': np.sum,
                                                                            'u_b2_count_in_6': np.sum,
                                                                            'u_b3_count_in_6': np.sum,
                                                                            'u_b4_count_in_6': np.sum})
df_part_1_u_b_count_in_6.reset_index(inplace = True)
df_part_1_u_b_count_in_6.head()                                                                        





df_part_1_u_b_count_in_6['u_b_count_in_6'] = df_part_1_u_b_count_in_6[['u_b1_count_in_6',
                                                                       'u_b2_count_in_6',
                                                                       'u_b3_count_in_6',
                                                                       'u_b4_count_in_6']].apply(lambda x: x.sum(), axis = 1)
df_part_1_u_b_count_in_6.head()



# u_b_count_in_3
df_part_1_in_3 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-25')]
df_part_1_in_3['cumcount'] = df_part_1_in_3.groupby(['user_id', 'behavior_type']).cumcount()
df_part_1_u_b_count_in_3 = df_part_1.drop_duplicates(['user_id','behavior_type'], 'last')[['user_id','behavior_type','cumcount']]
df_part_1_u_b_count_in_3 = pd.get_dummies(df_part_1_u_b_count_in_3['behavior_type']).join(df_part_1_u_b_count_in_3[['user_id','cumcount']])
df_part_1_u_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_u_b_count_in_3['u_b1_count_in_3'] = df_part_1_u_b_count_in_3['behavior_type_1'] * (df_part_1_u_b_count_in_3['cumcount']+1)
df_part_1_u_b_count_in_3['u_b2_count_in_3'] = df_part_1_u_b_count_in_3['behavior_type_2'] * (df_part_1_u_b_count_in_3['cumcount']+1)
df_part_1_u_b_count_in_3['u_b3_count_in_3'] = df_part_1_u_b_count_in_3['behavior_type_3'] * (df_part_1_u_b_count_in_3['cumcount']+1)
df_part_1_u_b_count_in_3['u_b4_count_in_3'] = df_part_1_u_b_count_in_3['behavior_type_4'] * (df_part_1_u_b_count_in_3['cumcount']+1)
df_part_1_u_b_count_in_3 = df_part_1_u_b_count_in_3.groupby('user_id').agg({'u_b1_count_in_3': np.sum,
                                                                            'u_b2_count_in_3': np.sum,
                                                                            'u_b3_count_in_3': np.sum,
                                                                            'u_b4_count_in_3': np.sum})
df_part_1_u_b_count_in_3.reset_index(inplace = True)
df_part_1_u_b_count_in_3['u_b_count_in_3'] = df_part_1_u_b_count_in_3[['u_b1_count_in_3',
                                                                       'u_b2_count_in_3',
                                                                       'u_b3_count_in_3',
                                                                       'u_b4_count_in_3']].apply(lambda x: x.sum(), axis = 1)
df_part_1_u_b_count_in_3.head()



# u_b_count_in_1
df_part_1_in_1 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-27')]
df_part_1_in_1['cumcount'] = df_part_1_in_1.groupby(['user_id', 'behavior_type']).cumcount()
df_part_1_u_b_count_in_1 = df_part_1_in_1.drop_duplicates(['user_id','behavior_type'], 'last')[['user_id','behavior_type','cumcount']]
df_part_1_u_b_count_in_1 = pd.get_dummies(df_part_1_u_b_count_in_1['behavior_type']).join(df_part_1_u_b_count_in_1[['user_id','cumcount']])
df_part_1_u_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_u_b_count_in_1['u_b1_count_in_1'] = df_part_1_u_b_count_in_1['behavior_type_1'] * (df_part_1_u_b_count_in_1['cumcount']+1)
df_part_1_u_b_count_in_1['u_b2_count_in_1'] = df_part_1_u_b_count_in_1['behavior_type_2'] * (df_part_1_u_b_count_in_1['cumcount']+1)
df_part_1_u_b_count_in_1['u_b3_count_in_1'] = df_part_1_u_b_count_in_1['behavior_type_3'] * (df_part_1_u_b_count_in_1['cumcount']+1)
df_part_1_u_b_count_in_1['u_b4_count_in_1'] = df_part_1_u_b_count_in_1['behavior_type_4'] * (df_part_1_u_b_count_in_1['cumcount']+1)
df_part_1_u_b_count_in_1 = df_part_1_u_b_count_in_1.groupby('user_id').agg({'u_b1_count_in_1': np.sum,
                                                                            'u_b2_count_in_1': np.sum,
                                                                            'u_b3_count_in_1': np.sum,
                                                                            'u_b4_count_in_1': np.sum})
df_part_1_u_b_count_in_1.reset_index(inplace = True)
df_part_1_u_b_count_in_1['u_b_count_in_1']  = df_part_1_u_b_count_in_1[['u_b1_count_in_1',
                                                                        'u_b2_count_in_1',
                                                                        'u_b3_count_in_1',
                                                                        'u_b4_count_in_1']].apply(lambda x: x.sum(), axis = 1)
df_part_1_u_b_count_in_1.head()



# merge the result of count_in_6, count_in_3, count_in_1

df_part_1_u_b_count = pd.merge(df_part_1_u_b_count_in_6, 
                               df_part_1_u_b_count_in_3, on = ['user_id'], how = 'left').fillna(0)
df_part_1_u_b_count.head()



df_part_1_u_b_count = pd.merge(df_part_1_u_b_count, 
                               df_part_1_u_b_count_in_1, on = ['user_id'], how = 'left').fillna(0)
df_part_1_u_b_count.head()


df_part_1_u_b_count[['u_b1_count_in_6',
                     'u_b2_count_in_6',
                     'u_b3_count_in_6',
                     'u_b4_count_in_6',
                      'u_b_count_in_6',
                     'u_b1_count_in_3',
                     'u_b2_count_in_3',
                     'u_b3_count_in_3',
                     'u_b4_count_in_3',
                      'u_b_count_in_3',
                     'u_b1_count_in_1',
                     'u_b2_count_in_1',
                     'u_b3_count_in_1',
                     'u_b4_count_in_1',
                      'u_b_count_in_1']] = df_part_1_u_b_count[['u_b1_count_in_6',
                                                                'u_b2_count_in_6',
                                                                'u_b3_count_in_6',
                                                                'u_b4_count_in_6',
                                                                 'u_b_count_in_6',
                                                                'u_b1_count_in_3',
                                                                'u_b2_count_in_3',
                                                                'u_b3_count_in_3',
                                                                'u_b4_count_in_3',
                                                                 'u_b_count_in_3',
                                                                'u_b1_count_in_1',
                                                                'u_b2_count_in_1',
                                                                'u_b3_count_in_1',
                                                                'u_b4_count_in_1',
                                                                 'u_b_count_in_1']].astype(int)
df_part_1_u_b_count.head()




# u_b4_rate
df_part_1_u_b_count['u_b4_rate'] = df_part_1_u_b_count['u_b4_count_in_6'] / df_part_1_u_b_count['u_b_count_in_6']
df_part_1_u_b_count.head()



# u_b4_diff_time
df_part_1 = df_part_1.sort_values(by = ['user_id', 'time'])
df_part_1.head(10)


df_part_1_u_b4_time = df_part_1[df_part_1['behavior_type'] == 4].drop_duplicates(['user_id'],'first')[['user_id','time']]
df_part_1_u_b4_time.columns = ['user_id','b4_first_time']
df_part_1_u_b4_time.head()


df_part_1_u_b_time = df_part_1.drop_duplicates(['user_id'],'first')[['user_id','time']]
df_part_1_u_b_time.columns = ['user_id','b_first_time']
df_part_1_u_b_time.head()

df_part_1_u_b_b4_time = pd.merge(df_part_1_u_b_time, df_part_1_u_b4_time, on = ['user_id'])
df_part_1_u_b_b4_time['u_b4_diff_time'] = df_part_1_u_b_b4_time['b4_first_time'] - df_part_1_u_b_b4_time['b_first_time']
df_part_1_u_b_b4_time = df_part_1_u_b_b4_time[['user_id', 'u_b4_diff_time']]
df_part_1_u_b_b4_time['u_b4_diff_hours'] = df_part_1_u_b_b4_time['u_b4_diff_time'].apply(lambda x: x.days * 24 + x.seconds//3600)
df_part_1_u_b_b4_time.head()


# generating feature set U
f_U_part_1 = pd.merge(df_part_1_u_b_count, 
                      df_part_1_u_b_b4_time, 
                      on = ['user_id'], how = 'left')[['user_id',
                                                       'u_b1_count_in_6', 
                                                       'u_b2_count_in_6', 
                                                       'u_b3_count_in_6', 
                                                       'u_b4_count_in_6', 
                                                       'u_b_count_in_6',
                                                       'u_b1_count_in_3',
                                                       'u_b2_count_in_3', 
                                                       'u_b3_count_in_3',
                                                       'u_b4_count_in_3', 
                                                       'u_b_count_in_3',
                                                       'u_b1_count_in_1',
                                                       'u_b2_count_in_1', 
                                                       'u_b3_count_in_1',
                                                       'u_b4_count_in_1', 
                                                       'u_b_count_in_1', 
                                                       'u_b4_rate', 
                                                       'u_b4_diff_hours']]
                      
# write to csv file
f_U_part_1 = f_U_part_1.round({'u_b4_rate': 3})
f_U_part_1.to_csv(path_df_part_1_U, index = False)


path_df = open(path_df_part_1, 'r')
df_part_1 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
df_part_1.columns = ['time','user_id','item_id','behavior_type','item_category']
df_part_1.head()


# i_u_count_in_6
df_part_1_in_6 = df_part_1.drop_duplicates(['item_id', 'user_id'])
df_part_1_in_6['i_u_count_in_6'] = df_part_1_in_6.groupby('item_id').cumcount() + 1
df_part_1_i_u_count_in_6 = df_part_1_in_6.drop_duplicates(['item_id'], 'last')[['item_id', 'i_u_count_in_6']]
df_part_1_i_u_count_in_6.head()


# i_u_count_in_3
df_part_1_in_3 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-25')].drop_duplicates(['item_id', 'user_id'])
df_part_1_in_3['i_u_count_in_3'] = df_part_1_in_3.groupby('item_id').cumcount() + 1
df_part_1_i_u_count_in_3 = df_part_1_in_3.drop_duplicates(['item_id'], 'last')[['item_id', 'i_u_count_in_3']]
df_part_1_i_u_count_in_3.head()


# i_u_count_in_1
df_part_1_in_1 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-27')].drop_duplicates(['item_id', 'user_id'])
df_part_1_in_1['i_u_count_in_1'] = df_part_1_in_1.groupby('item_id').cumcount() + 1
df_part_1_i_u_count_in_1 = df_part_1_in_1.drop_duplicates(['item_id'], 'last')[['item_id', 'i_u_count_in_1']]


# merge for generation of i_u_count
df_part_1_i_u_count = pd.merge(df_part_1_i_u_count_in_6, 
                               df_part_1_i_u_count_in_3,
                               on=['item_id'],how='left').fillna(0)
df_part_1_i_u_count = pd.merge(df_part_1_i_u_count, 
                               df_part_1_i_u_count_in_1,
                               on=['item_id'],how='left').fillna(0)
df_part_1_i_u_count[['i_u_count_in_6',
                     'i_u_count_in_3',
                     'i_u_count_in_1']] = df_part_1_i_u_count[['i_u_count_in_6',
                                                               'i_u_count_in_3',
                                                               'i_u_count_in_1']].astype(int)
df_part_1_i_u_count.head()


# i_b_count_in_6
df_part_1['cumcount'] = df_part_1.groupby(['item_id', 'behavior_type']).cumcount()
df_part_1_i_b_count_in_6 = df_part_1.drop_duplicates(['item_id','behavior_type'], 'last')[['item_id','behavior_type','cumcount']]
df_part_1_i_b_count_in_6 = pd.get_dummies(df_part_1_i_b_count_in_6['behavior_type']).join(df_part_1_i_b_count_in_6[['item_id','cumcount']])
df_part_1_i_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_i_b_count_in_6['i_b1_count_in_6'] = df_part_1_i_b_count_in_6['behavior_type_1'] * (df_part_1_i_b_count_in_6['cumcount']+1)
df_part_1_i_b_count_in_6['i_b2_count_in_6'] = df_part_1_i_b_count_in_6['behavior_type_2'] * (df_part_1_i_b_count_in_6['cumcount']+1)
df_part_1_i_b_count_in_6['i_b3_count_in_6'] = df_part_1_i_b_count_in_6['behavior_type_3'] * (df_part_1_i_b_count_in_6['cumcount']+1)
df_part_1_i_b_count_in_6['i_b4_count_in_6'] = df_part_1_i_b_count_in_6['behavior_type_4'] * (df_part_1_i_b_count_in_6['cumcount']+1)
df_part_1_i_b_count_in_6 = df_part_1_i_b_count_in_6[['item_id', 
                                                     'i_b1_count_in_6', 
                                                     'i_b2_count_in_6', 
                                                     'i_b3_count_in_6',
                                                     'i_b4_count_in_6']]
df_part_1_i_b_count_in_6 = df_part_1_i_b_count_in_6.groupby('item_id').agg({'i_b1_count_in_6': np.sum,
                                                                            'i_b2_count_in_6': np.sum,
                                                                            'i_b3_count_in_6': np.sum,
                                                                            'i_b4_count_in_6': np.sum})
df_part_1_i_b_count_in_6.reset_index(inplace = True)
df_part_1_i_b_count_in_6['i_b_count_in_6'] = df_part_1_i_b_count_in_6['i_b1_count_in_6'] +                                              df_part_1_i_b_count_in_6['i_b2_count_in_6'] +                                              df_part_1_i_b_count_in_6['i_b3_count_in_6'] +                                              df_part_1_i_b_count_in_6['i_b4_count_in_6']

df_part_1_i_b_count_in_6.head()


# i_b_count_in_3
df_part_1_in_3 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-25')]
df_part_1_in_3['cumcount'] = df_part_1_in_3.groupby(['item_id', 'behavior_type']).cumcount()
df_part_1_i_b_count_in_3 = df_part_1.drop_duplicates(['item_id','behavior_type'], 'last')[['item_id','behavior_type','cumcount']]
df_part_1_i_b_count_in_3 = pd.get_dummies(df_part_1_i_b_count_in_3['behavior_type']).join(df_part_1_i_b_count_in_3[['item_id','cumcount']])
df_part_1_i_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_i_b_count_in_3['i_b1_count_in_3'] = df_part_1_i_b_count_in_3['behavior_type_1'] * (df_part_1_i_b_count_in_3['cumcount']+1)
df_part_1_i_b_count_in_3['i_b2_count_in_3'] = df_part_1_i_b_count_in_3['behavior_type_2'] * (df_part_1_i_b_count_in_3['cumcount']+1)
df_part_1_i_b_count_in_3['i_b3_count_in_3'] = df_part_1_i_b_count_in_3['behavior_type_3'] * (df_part_1_i_b_count_in_3['cumcount']+1)
df_part_1_i_b_count_in_3['i_b4_count_in_3'] = df_part_1_i_b_count_in_3['behavior_type_4'] * (df_part_1_i_b_count_in_3['cumcount']+1)
df_part_1_i_b_count_in_3 = df_part_1_i_b_count_in_3[['item_id', 
                                                     'i_b1_count_in_3', 
                                                     'i_b2_count_in_3', 
                                                     'i_b3_count_in_3',
                                                     'i_b4_count_in_3']]
df_part_1_i_b_count_in_3 = df_part_1_i_b_count_in_3.groupby('item_id').agg({'i_b1_count_in_3': np.sum,
                                                                            'i_b2_count_in_3': np.sum,
                                                                            'i_b3_count_in_3': np.sum,
                                                                            'i_b4_count_in_3': np.sum})
df_part_1_i_b_count_in_3.reset_index(inplace = True)
df_part_1_i_b_count_in_3['i_b_count_in_3'] = df_part_1_i_b_count_in_3['i_b1_count_in_3'] +                                              df_part_1_i_b_count_in_3['i_b2_count_in_3'] +                                              df_part_1_i_b_count_in_3['i_b3_count_in_3'] +                                              df_part_1_i_b_count_in_3['i_b4_count_in_3']


# i_b_count_in_1
df_part_1_in_1 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-27')]
df_part_1_in_1['cumcount'] = df_part_1_in_1.groupby(['item_id', 'behavior_type']).cumcount()
df_part_1_i_b_count_in_1 = df_part_1_in_1.drop_duplicates(['item_id','behavior_type'], 'last')[['item_id','behavior_type','cumcount']]
df_part_1_i_b_count_in_1 = pd.get_dummies(df_part_1_i_b_count_in_1['behavior_type']).join(df_part_1_i_b_count_in_1[['item_id','cumcount']])
df_part_1_i_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)                     
df_part_1_i_b_count_in_1['i_b1_count_in_1'] = df_part_1_i_b_count_in_1['behavior_type_1'] * (df_part_1_i_b_count_in_1['cumcount']+1)
df_part_1_i_b_count_in_1['i_b2_count_in_1'] = df_part_1_i_b_count_in_1['behavior_type_2'] * (df_part_1_i_b_count_in_1['cumcount']+1)
df_part_1_i_b_count_in_1['i_b3_count_in_1'] = df_part_1_i_b_count_in_1['behavior_type_3'] * (df_part_1_i_b_count_in_1['cumcount']+1)
df_part_1_i_b_count_in_1['i_b4_count_in_1'] = df_part_1_i_b_count_in_1['behavior_type_4'] * (df_part_1_i_b_count_in_1['cumcount']+1)
df_part_1_i_b_count_in_1 = df_part_1_i_b_count_in_1[['item_id', 
                                                     'i_b1_count_in_1', 
                                                     'i_b2_count_in_1', 
                                                     'i_b3_count_in_1',
                                                     'i_b4_count_in_1']]
df_part_1_i_b_count_in_1 = df_part_1_i_b_count_in_1.groupby('item_id').agg({'i_b1_count_in_1': np.sum,
                                                                            'i_b2_count_in_1': np.sum,
                                                                            'i_b3_count_in_1': np.sum,
                                                                            'i_b4_count_in_1': np.sum})
df_part_1_i_b_count_in_1.reset_index(inplace = True)
df_part_1_i_b_count_in_1['i_b_count_in_1'] = df_part_1_i_b_count_in_1['i_b1_count_in_1'] +                                              df_part_1_i_b_count_in_1['i_b2_count_in_1'] +                                              df_part_1_i_b_count_in_1['i_b3_count_in_1'] +                                              df_part_1_i_b_count_in_1['i_b4_count_in_1']


# merge for generation of i_b_count
df_part_1_i_b_count = pd.merge(df_part_1_i_b_count_in_6, 
                               df_part_1_i_b_count_in_3, 
                               on = ['item_id'], how = 'left').fillna(0)
df_part_1_i_b_count = pd.merge(df_part_1_i_b_count, 
                               df_part_1_i_b_count_in_1, 
                               on = ['item_id'], how = 'left').fillna(0)
df_part_1_i_b_count[['i_b1_count_in_6',
                     'i_b2_count_in_6',
                     'i_b3_count_in_6',
                     'i_b4_count_in_6',
                      'i_b_count_in_6',
                     'i_b1_count_in_3',
                     'i_b2_count_in_3',
                     'i_b3_count_in_3',
                     'i_b4_count_in_3',
                      'i_b_count_in_3',
                     'i_b1_count_in_1',
                     'i_b2_count_in_1',
                     'i_b3_count_in_1',
                     'i_b4_count_in_1',
                      'i_b_count_in_1']] = df_part_1_i_b_count[['i_b1_count_in_6',
                                                                'i_b2_count_in_6',
                                                                'i_b3_count_in_6',
                                                                'i_b4_count_in_6',
                                                                 'i_b_count_in_6',
                                                                'i_b1_count_in_3',
                                                                'i_b2_count_in_3',
                                                                'i_b3_count_in_3',
                                                                'i_b4_count_in_3',
                                                                 'i_b_count_in_3',
                                                                'i_b1_count_in_1',
                                                                'i_b2_count_in_1',
                                                                'i_b3_count_in_1',
                                                                'i_b4_count_in_1',
                                                                 'i_b_count_in_1']].astype(int)
df_part_1_i_b_count.head()


# i_b4_rate
df_part_1_i_b_count['i_b4_rate'] = df_part_1_i_b_count['i_b4_count_in_6'] / df_part_1_i_b_count['i_b_count_in_6']
df_part_1_i_b_count.head()


# i_b4_diff_time
df_part_1 = df_part_1.sort_values(by=['item_id', 'time'])
df_part_1_i_b4_time = df_part_1[df_part_1['behavior_type'] == 4].drop_duplicates(['item_id'], 'first')[['item_id','time']]
df_part_1_i_b4_time.columns = ['item_id','b4_first_time']
df_part_1_i_b_time = df_part_1.drop_duplicates(['item_id'], 'first')[['item_id','time']]
df_part_1_i_b_time.columns = ['item_id','b_first_time']
df_part_1_i_b_b4_time = pd.merge(df_part_1_i_b_time, df_part_1_i_b4_time, on = ['item_id'])
df_part_1_i_b_b4_time['i_b4_diff_time']  = df_part_1_i_b_b4_time['b4_first_time'] - df_part_1_i_b_b4_time['b_first_time']
df_part_1_i_b_b4_time['i_b4_diff_hours'] = df_part_1_i_b_b4_time['i_b4_diff_time'].apply(lambda x: x.days * 24 + x.seconds//3600)
df_part_1_i_b_b4_time = df_part_1_i_b_b4_time[['item_id', 'i_b4_diff_hours']]
df_part_1_i_b_b4_time.head()


# generating feature set I
f_I_part_1 = pd.merge(df_part_1_i_b_count, 
                      df_part_1_i_b_b4_time, 
                      on = ['item_id'], how = 'left')
f_I_part_1 = pd.merge(f_I_part_1, 
                      df_part_1_i_u_count, 
                      on = ['item_id'], how = 'left')[['item_id', 
                                                       'i_u_count_in_6', 
                                                       'i_u_count_in_3', 
                                                       'i_u_count_in_1',
                                                       'i_b1_count_in_6', 
                                                       'i_b2_count_in_6', 
                                                       'i_b3_count_in_6', 
                                                       'i_b4_count_in_6', 
                                                       'i_b_count_in_6', 
                                                       'i_b1_count_in_3',
                                                       'i_b2_count_in_3',
                                                       'i_b3_count_in_3',
                                                       'i_b4_count_in_3',
                                                       'i_b_count_in_3',
                                                       'i_b1_count_in_1', 
                                                       'i_b2_count_in_1', 
                                                       'i_b3_count_in_1', 
                                                       'i_b4_count_in_1', 
                                                       'i_b_count_in_1',
                                                       'i_b4_rate', 
                                                       'i_b4_diff_hours']]
                      
# write to csv file
f_I_part_1 = f_I_part_1.round({'i_b4_rate': 3})
f_I_part_1.to_csv(path_df_part_1_I, index = False)


# loading data
path_df = open(path_df_part_1, 'r')
df_part_1 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
df_part_1.columns = ['time','user_id','item_id','behavior_type','item_category']
# c_u_count_in_6
df_part_1_in_6 = df_part_1.drop_duplicates(['item_category', 'user_id'])
df_part_1_in_6['c_u_count_in_6'] = df_part_1_in_6.groupby('item_category').cumcount() + 1
df_part_1_c_u_count_in_6 = df_part_1_in_6.drop_duplicates(['item_category'], 'last')[['item_category', 'c_u_count_in_6']]

# c_u_count_in_3
df_part_1_in_3 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-25')].drop_duplicates(['item_category', 'user_id'])
df_part_1_in_3['c_u_count_in_3'] = df_part_1_in_3.groupby('item_category').cumcount() + 1
df_part_1_c_u_count_in_3 = df_part_1_in_3.drop_duplicates(['item_category'], 'last')[['item_category', 'c_u_count_in_3']]

# c_u_count_in_1
df_part_1_in_1 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-27')].drop_duplicates(['item_category', 'user_id'])
df_part_1_in_1['c_u_count_in_1'] = df_part_1_in_1.groupby('item_category').cumcount() + 1
df_part_1_c_u_count_in_1 = df_part_1_in_1.drop_duplicates(['item_category'], 'last')[['item_category', 'c_u_count_in_1']]

df_part_1_c_u_count = pd.merge(df_part_1_c_u_count_in_6, df_part_1_c_u_count_in_3,on=['item_category'],how='left').fillna(0)
df_part_1_c_u_count = pd.merge(df_part_1_c_u_count, df_part_1_c_u_count_in_1,on=['item_category'],how='left').fillna(0)
df_part_1_c_u_count[['c_u_count_in_6',
                     'c_u_count_in_3',
                     'c_u_count_in_1']] = df_part_1_c_u_count[['c_u_count_in_6',
                                                               'c_u_count_in_3',
                                                               'c_u_count_in_1']].astype(int)




# c_b_count_in_6
df_part_1['cumcount'] = df_part_1.groupby(['item_category', 'behavior_type']).cumcount()
df_part_1_c_b_count_in_6 = df_part_1.drop_duplicates(['item_category','behavior_type'], 'last')[['item_category','behavior_type','cumcount']]
df_part_1_c_b_count_in_6 = pd.get_dummies(df_part_1_c_b_count_in_6['behavior_type']).join(df_part_1_c_b_count_in_6[['item_category','cumcount']])
df_part_1_c_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_c_b_count_in_6['c_b1_count_in_6'] = df_part_1_c_b_count_in_6['behavior_type_1'] * (df_part_1_c_b_count_in_6['cumcount']+1)
df_part_1_c_b_count_in_6['c_b2_count_in_6'] = df_part_1_c_b_count_in_6['behavior_type_2'] * (df_part_1_c_b_count_in_6['cumcount']+1)
df_part_1_c_b_count_in_6['c_b3_count_in_6'] = df_part_1_c_b_count_in_6['behavior_type_3'] * (df_part_1_c_b_count_in_6['cumcount']+1)
df_part_1_c_b_count_in_6['c_b4_count_in_6'] = df_part_1_c_b_count_in_6['behavior_type_4'] * (df_part_1_c_b_count_in_6['cumcount']+1)
df_part_1_c_b_count_in_6 = df_part_1_c_b_count_in_6[['item_category', 
                                                     'c_b1_count_in_6', 
                                                     'c_b2_count_in_6', 
                                                     'c_b3_count_in_6',
                                                     'c_b4_count_in_6']]
df_part_1_c_b_count_in_6 = df_part_1_c_b_count_in_6.groupby('item_category').agg({'c_b1_count_in_6': np.sum,
                                                                                  'c_b2_count_in_6': np.sum,
                                                                                  'c_b3_count_in_6': np.sum,
                                                                                  'c_b4_count_in_6': np.sum})
df_part_1_c_b_count_in_6.reset_index(inplace = True)
df_part_1_c_b_count_in_6['c_b_count_in_6'] = df_part_1_c_b_count_in_6['c_b1_count_in_6'] +                                              df_part_1_c_b_count_in_6['c_b2_count_in_6'] +                                              df_part_1_c_b_count_in_6['c_b3_count_in_6'] +                                              df_part_1_c_b_count_in_6['c_b4_count_in_6']





# c_b_count_in_3
df_part_1_in_3 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-25')]
df_part_1_in_3['cumcount'] = df_part_1_in_3.groupby(['item_category', 'behavior_type']).cumcount()
df_part_1_c_b_count_in_3 = df_part_1_in_3.drop_duplicates(['item_category','behavior_type'], 'last')[['item_category','behavior_type','cumcount']]
df_part_1_c_b_count_in_3 = pd.get_dummies(df_part_1_c_b_count_in_3['behavior_type']).join(df_part_1_c_b_count_in_3[['item_category','cumcount']])
df_part_1_c_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_c_b_count_in_3['c_b1_count_in_3'] = df_part_1_c_b_count_in_3['behavior_type_1'] * (df_part_1_c_b_count_in_3['cumcount']+1)
df_part_1_c_b_count_in_3['c_b2_count_in_3'] = df_part_1_c_b_count_in_3['behavior_type_2'] * (df_part_1_c_b_count_in_3['cumcount']+1)
df_part_1_c_b_count_in_3['c_b3_count_in_3'] = df_part_1_c_b_count_in_3['behavior_type_3'] * (df_part_1_c_b_count_in_3['cumcount']+1)
df_part_1_c_b_count_in_3['c_b4_count_in_3'] = df_part_1_c_b_count_in_3['behavior_type_4'] * (df_part_1_c_b_count_in_3['cumcount']+1)
df_part_1_c_b_count_in_3 = df_part_1_c_b_count_in_3[['item_category', 
                                                     'c_b1_count_in_3', 
                                                     'c_b2_count_in_3', 
                                                     'c_b3_count_in_3',
                                                     'c_b4_count_in_3']]
df_part_1_c_b_count_in_3 = df_part_1_c_b_count_in_3.groupby('item_category').agg({'c_b1_count_in_3': np.sum,
                                                                                  'c_b2_count_in_3': np.sum,
                                                                                  'c_b3_count_in_3': np.sum,
                                                                                  'c_b4_count_in_3': np.sum})
df_part_1_c_b_count_in_3.reset_index(inplace = True)
df_part_1_c_b_count_in_3['c_b_count_in_3'] = df_part_1_c_b_count_in_3['c_b1_count_in_3'] +                                              df_part_1_c_b_count_in_3['c_b2_count_in_3'] +                                              df_part_1_c_b_count_in_3['c_b3_count_in_3'] +                                              df_part_1_c_b_count_in_3['c_b4_count_in_3']




# c_b_count_in_1
df_part_1_in_1 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-27')]
df_part_1_in_1['cumcount'] = df_part_1_in_1.groupby(['item_category', 'behavior_type']).cumcount()
df_part_1_c_b_count_in_1 = df_part_1_in_1.drop_duplicates(['item_category','behavior_type'], 'last')[['item_category','behavior_type','cumcount']]
df_part_1_c_b_count_in_1 = pd.get_dummies(df_part_1_c_b_count_in_1['behavior_type']).join(df_part_1_c_b_count_in_1[['item_category','cumcount']])
df_part_1_c_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_1_c_b_count_in_1['c_b1_count_in_1'] = df_part_1_c_b_count_in_1['behavior_type_1'] * (df_part_1_c_b_count_in_1['cumcount']+1)
df_part_1_c_b_count_in_1['c_b2_count_in_1'] = df_part_1_c_b_count_in_1['behavior_type_2'] * (df_part_1_c_b_count_in_1['cumcount']+1)
df_part_1_c_b_count_in_1['c_b3_count_in_1'] = df_part_1_c_b_count_in_1['behavior_type_3'] * (df_part_1_c_b_count_in_1['cumcount']+1)
df_part_1_c_b_count_in_1['c_b4_count_in_1'] = df_part_1_c_b_count_in_1['behavior_type_4'] * (df_part_1_c_b_count_in_1['cumcount']+1)
df_part_1_c_b_count_in_1 = df_part_1_c_b_count_in_1[['item_category', 
                                                     'c_b1_count_in_1', 
                                                     'c_b2_count_in_1', 
                                                     'c_b3_count_in_1',
                                                     'c_b4_count_in_1']]
df_part_1_c_b_count_in_1 = df_part_1_c_b_count_in_1.groupby('item_category').agg({'c_b1_count_in_1': np.sum,
                                                                                  'c_b2_count_in_1': np.sum,
                                                                                  'c_b3_count_in_1': np.sum,
                                                                                  'c_b4_count_in_1': np.sum})
df_part_1_c_b_count_in_1.reset_index(inplace = True)
df_part_1_c_b_count_in_1['c_b_count_in_1'] = df_part_1_c_b_count_in_1['c_b1_count_in_1'] +                                              df_part_1_c_b_count_in_1['c_b2_count_in_1'] +                                              df_part_1_c_b_count_in_1['c_b3_count_in_1'] +                                              df_part_1_c_b_count_in_1['c_b4_count_in_1']    
                                             
df_part_1_c_b_count = pd.merge(df_part_1_c_b_count_in_6, df_part_1_c_b_count_in_3, on = ['item_category'], how = 'left').fillna(0)                                      
df_part_1_c_b_count = pd.merge(df_part_1_c_b_count, df_part_1_c_b_count_in_1, on = ['item_category'], how = 'left').fillna(0)
df_part_1_c_b_count[['c_b1_count_in_6',
                     'c_b2_count_in_6',
                     'c_b3_count_in_6',
                     'c_b4_count_in_6',
                      'c_b_count_in_6',
                     'c_b1_count_in_3',
                     'c_b2_count_in_3',
                     'c_b3_count_in_3',
                     'c_b4_count_in_3',
                      'c_b_count_in_3',
                     'c_b1_count_in_1',
                     'c_b2_count_in_1',
                     'c_b3_count_in_1',
                     'c_b4_count_in_1',
                      'c_b_count_in_1']] = df_part_1_c_b_count[['c_b1_count_in_6',
                                                                'c_b2_count_in_6',
                                                                'c_b3_count_in_6',
                                                                'c_b4_count_in_6',
                                                                 'c_b_count_in_6',
                                                                'c_b1_count_in_3',
                                                                'c_b2_count_in_3',
                                                                'c_b3_count_in_3',
                                                                'c_b4_count_in_3',
                                                                 'c_b_count_in_3',
                                                                'c_b1_count_in_1',
                                                                'c_b2_count_in_1',
                                                                'c_b3_count_in_1',
                                                                'c_b4_count_in_1',
                                                                 'c_b_count_in_1']].astype(int)




# c_b4_rate
df_part_1_c_b_count['c_b4_rate'] = df_part_1_c_b_count['c_b4_count_in_6'] / df_part_1_c_b_count['c_b_count_in_6']





# c_b4_diff_time
df_part_1 = df_part_1.sort_values(by=['item_category', 'time'])
df_part_1_c_b4_time = df_part_1[df_part_1['behavior_type'] == 4].drop_duplicates(['item_category'], 'first')[['item_category','time']]
df_part_1_c_b4_time.columns = ['item_category','b4_first_time']
df_part_1_c_b_time = df_part_1.drop_duplicates(['item_category'], 'first')[['item_category','time']]
df_part_1_c_b_time.columns = ['item_category','b_first_time']
df_part_1_c_b_b4_time = pd.merge(df_part_1_c_b_time, df_part_1_c_b4_time, on = ['item_category'])
df_part_1_c_b_b4_time['c_b4_diff_time']  = df_part_1_c_b_b4_time['b4_first_time'] - df_part_1_c_b_b4_time['b_first_time']
df_part_1_c_b_b4_time['c_b4_diff_hours'] = df_part_1_c_b_b4_time['c_b4_diff_time'].apply(lambda x: x.days * 24 + x.seconds//3600)
df_part_1_c_b_b4_time = df_part_1_c_b_b4_time[['item_category',
                                               'c_b4_diff_hours']]




# generating feature set C
f_C_part_1 = pd.merge(df_part_1_c_u_count, df_part_1_c_b_count, on = ['item_category'], how = 'left')
f_C_part_1 = pd.merge(f_C_part_1, df_part_1_c_b_b4_time, on = ['item_category'], how = 'left')
f_C_part_1 = f_C_part_1.round({'c_b4_rate': 3})

# write to csv file
f_C_part_1.to_csv(path_df_part_1_C, index = False)




# get df_part_1_i_ub_count
path_df = open(path_df_part_1_I, 'r')
df_part_1_I = pd.read_csv(path_df, index_col = False)
df_part_1_I.head()



df_part_1_i_ub_count = df_part_1_I[['item_id','i_u_count_in_6','i_b_count_in_6','i_b4_count_in_6']]
df_part_1_i_ub_count.head()



path_df = open(path_df_part_1_uic_label, 'r')
df_part_1_uic = pd.read_csv(path_df, index_col = False)
df_part_1_uic.head()


# In[55]:


df_part_1_ic_u_b_count = pd.merge(df_part_1_uic, df_part_1_i_ub_count, on=['item_id'], how='left').fillna(0)
df_part_1_ic_u_b_count = df_part_1_ic_u_b_count.drop_duplicates(['item_id','item_category'])
df_part_1_ic_u_b_count.head()


# In[56]:


# ic_u_rank_in_c
df_part_1_ic_u_b_count['ic_u_rank_in_c'] = df_part_1_ic_u_b_count.groupby('item_category')['i_u_count_in_6'].rank(method='min',ascending=False).astype('int')
# ic_b_rank_in_c
df_part_1_ic_u_b_count['ic_b_rank_in_c'] = df_part_1_ic_u_b_count.groupby('item_category')['i_b_count_in_6'].rank(method='min',ascending=False).astype('int')
# ic_b4_rank_in_c
df_part_1_ic_u_b_count['ic_b4_rank_in_c'] = df_part_1_ic_u_b_count.groupby('item_category')['i_b4_count_in_6'].rank(method='min',ascending=False).astype('int')

f_IC_part_1 = df_part_1_ic_u_b_count[['item_id', 
                                      'item_category', 
                                      'ic_u_rank_in_c', 
                                      'ic_b_rank_in_c', 
                                      'ic_b4_rank_in_c']]
# write to csv file
f_IC_part_1.to_csv(path_df_part_1_IC, index = False)


# In[57]:


df_part_1_ic_u_b_count.head()


# In[58]:


path_df = open(path_df_part_1, 'r')
df_part_1 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
df_part_1.columns = ['time','user_id','item_id','behavior_type','item_category']


# In[59]:


# ui_b_count_in_6
df_part_1['cumcount'] = df_part_1.groupby(['user_id', 'item_id', 'behavior_type']).cumcount()
df_part_1_ui_b_count_in_6 = df_part_1.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','cumcount']]
df_part_1_ui_b_count_in_6 = pd.get_dummies(df_part_1_ui_b_count_in_6['behavior_type']).join(df_part_1_ui_b_count_in_6[['user_id','item_id','cumcount']])
df_part_1_ui_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_1_ui_b_count_in_6['ui_b1_count_in_6'] = df_part_1_ui_b_count_in_6['behavior_type_1'] * (df_part_1_ui_b_count_in_6['cumcount']+1)
df_part_1_ui_b_count_in_6['ui_b2_count_in_6'] = df_part_1_ui_b_count_in_6['behavior_type_2'] * (df_part_1_ui_b_count_in_6['cumcount']+1)
df_part_1_ui_b_count_in_6['ui_b3_count_in_6'] = df_part_1_ui_b_count_in_6['behavior_type_3'] * (df_part_1_ui_b_count_in_6['cumcount']+1)
df_part_1_ui_b_count_in_6['ui_b4_count_in_6'] = df_part_1_ui_b_count_in_6['behavior_type_4'] * (df_part_1_ui_b_count_in_6['cumcount']+1)
df_part_1_ui_b_count_in_6 = df_part_1_ui_b_count_in_6[['user_id', 
                                                       'item_id', 
                                                       'ui_b1_count_in_6', 
                                                       'ui_b2_count_in_6', 
                                                       'ui_b3_count_in_6',
                                                       'ui_b4_count_in_6']]
df_part_1_ui_b_count_in_6 = df_part_1_ui_b_count_in_6.groupby(['user_id', 'item_id']).agg({'ui_b1_count_in_6': np.sum,
                                                                                           'ui_b2_count_in_6': np.sum,
                                                                                           'ui_b3_count_in_6': np.sum,
                                                                                           'ui_b4_count_in_6': np.sum})
df_part_1_ui_b_count_in_6.reset_index(inplace = True)
df_part_1_ui_b_count_in_6['ui_b_count_in_6'] = df_part_1_ui_b_count_in_6['ui_b1_count_in_6'] +                                                df_part_1_ui_b_count_in_6['ui_b2_count_in_6'] +                                                df_part_1_ui_b_count_in_6['ui_b3_count_in_6'] +                                                df_part_1_ui_b_count_in_6['ui_b4_count_in_6']
df_part_1_ui_b_count_in_6.head()




# In[60]:


# ui_b_count_in_3
df_part_1_in_3 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-25')]
df_part_1_in_3['cumcount'] = df_part_1_in_3.groupby(['user_id', 'item_id', 'behavior_type']).cumcount()
df_part_1_ui_b_count_in_3 = df_part_1.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','cumcount']]
df_part_1_ui_b_count_in_3 = pd.get_dummies(df_part_1_ui_b_count_in_3['behavior_type']).join(df_part_1_ui_b_count_in_3[['user_id','item_id','cumcount']])
df_part_1_ui_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_1_ui_b_count_in_3['ui_b1_count_in_3'] = df_part_1_ui_b_count_in_3['behavior_type_1'] * (df_part_1_ui_b_count_in_3['cumcount']+1)
df_part_1_ui_b_count_in_3['ui_b2_count_in_3'] = df_part_1_ui_b_count_in_3['behavior_type_2'] * (df_part_1_ui_b_count_in_3['cumcount']+1)
df_part_1_ui_b_count_in_3['ui_b3_count_in_3'] = df_part_1_ui_b_count_in_3['behavior_type_3'] * (df_part_1_ui_b_count_in_3['cumcount']+1)
df_part_1_ui_b_count_in_3['ui_b4_count_in_3'] = df_part_1_ui_b_count_in_3['behavior_type_4'] * (df_part_1_ui_b_count_in_3['cumcount']+1)
df_part_1_ui_b_count_in_3 = df_part_1_ui_b_count_in_3[['user_id', 
                                                       'item_id', 
                                                       'ui_b1_count_in_3', 
                                                       'ui_b2_count_in_3', 
                                                       'ui_b3_count_in_3',
                                                       'ui_b4_count_in_3']]
df_part_1_ui_b_count_in_3 = df_part_1_ui_b_count_in_3.groupby(['user_id', 'item_id']).agg({'ui_b1_count_in_3': np.sum,
                                                                                           'ui_b2_count_in_3': np.sum,
                                                                                           'ui_b3_count_in_3': np.sum,
                                                                                           'ui_b4_count_in_3': np.sum})
df_part_1_ui_b_count_in_3.reset_index(inplace = True)
df_part_1_ui_b_count_in_3['ui_b_count_in_3'] = df_part_1_ui_b_count_in_3['ui_b1_count_in_3'] +                                                df_part_1_ui_b_count_in_3['ui_b2_count_in_3'] +                                                df_part_1_ui_b_count_in_3['ui_b3_count_in_3'] +                                                df_part_1_ui_b_count_in_3['ui_b4_count_in_3']


# In[61]:


# ui_b_count_in_1
df_part_1_in_1 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-27')]
df_part_1_in_1['cumcount'] = df_part_1_in_1.groupby(['user_id', 'item_id', 'behavior_type']).cumcount()
df_part_1_ui_b_count_in_1 = df_part_1_in_1.drop_duplicates(['user_id','item_id','behavior_type'], 'last')[['user_id','item_id','behavior_type','cumcount']]
df_part_1_ui_b_count_in_1 = pd.get_dummies(df_part_1_ui_b_count_in_1['behavior_type']).join(df_part_1_ui_b_count_in_1[['user_id','item_id','cumcount']])
df_part_1_ui_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)
df_part_1_ui_b_count_in_1['ui_b1_count_in_1'] = df_part_1_ui_b_count_in_1['behavior_type_1'] * (df_part_1_ui_b_count_in_1['cumcount']+1)
df_part_1_ui_b_count_in_1['ui_b2_count_in_1'] = df_part_1_ui_b_count_in_1['behavior_type_2'] * (df_part_1_ui_b_count_in_1['cumcount']+1)
df_part_1_ui_b_count_in_1['ui_b3_count_in_1'] = df_part_1_ui_b_count_in_1['behavior_type_3'] * (df_part_1_ui_b_count_in_1['cumcount']+1)
df_part_1_ui_b_count_in_1['ui_b4_count_in_1'] = df_part_1_ui_b_count_in_1['behavior_type_4'] * (df_part_1_ui_b_count_in_1['cumcount']+1)
df_part_1_ui_b_count_in_1 = df_part_1_ui_b_count_in_1[['user_id',
                                                       'item_id', 
                                                       'ui_b1_count_in_1', 
                                                       'ui_b2_count_in_1', 
                                                       'ui_b3_count_in_1',
                                                       'ui_b4_count_in_1']]
df_part_1_ui_b_count_in_1 = df_part_1_ui_b_count_in_1.groupby(['user_id', 'item_id']).agg({'ui_b1_count_in_1': np.sum,
                                                                                           'ui_b2_count_in_1': np.sum,
                                                                                           'ui_b3_count_in_1': np.sum,
                                                                                           'ui_b4_count_in_1': np.sum})
df_part_1_ui_b_count_in_1.reset_index(inplace = True)
df_part_1_ui_b_count_in_1['ui_b_count_in_1'] = df_part_1_ui_b_count_in_1['ui_b1_count_in_1'] +                                                df_part_1_ui_b_count_in_1['ui_b2_count_in_1'] +                                                df_part_1_ui_b_count_in_1['ui_b3_count_in_1'] +                                                df_part_1_ui_b_count_in_1['ui_b4_count_in_1']


# In[62]:


df_part_1_ui_b_count = pd.merge(df_part_1_ui_b_count_in_6, df_part_1_ui_b_count_in_3, on = ['user_id','item_id'], how = 'left').fillna(0)
df_part_1_ui_b_count = pd.merge(df_part_1_ui_b_count, df_part_1_ui_b_count_in_1, on = ['user_id','item_id'], how = 'left').fillna(0)
df_part_1_ui_b_count[['ui_b1_count_in_6',
                      'ui_b2_count_in_6',
                      'ui_b3_count_in_6',
                      'ui_b4_count_in_6',
                       'ui_b_count_in_6',
                      'ui_b1_count_in_3',
                      'ui_b2_count_in_3',
                      'ui_b3_count_in_3',
                      'ui_b4_count_in_3',
                       'ui_b_count_in_3',
                      'ui_b1_count_in_1',
                      'ui_b2_count_in_1',
                      'ui_b3_count_in_1',
                      'ui_b4_count_in_1',
                       'ui_b_count_in_1']] = df_part_1_ui_b_count[['ui_b1_count_in_6',
                                                                   'ui_b2_count_in_6',
                                                                   'ui_b3_count_in_6',
                                                                   'ui_b4_count_in_6',
                                                                    'ui_b_count_in_6',
                                                                   'ui_b1_count_in_3',
                                                                   'ui_b2_count_in_3',
                                                                   'ui_b3_count_in_3',
                                                                   'ui_b4_count_in_3',
                                                                    'ui_b_count_in_3',
                                                                   'ui_b1_count_in_1',
                                                                   'ui_b2_count_in_1',
                                                                   'ui_b3_count_in_1',
                                                                   'ui_b4_count_in_1',
                                                                    'ui_b_count_in_1']].astype(int)


# In[63]:


df_part_1_ui_b_count.head()


# In[64]:


# ui_b_count_rank_in_u
df_part_1_ui_b_count['ui_b_count_rank_in_u'] = df_part_1_ui_b_count.groupby(['user_id'])['ui_b_count_in_6'].rank(method='min',ascending=False).astype('int')
df_part_1_ui_b_count.head()


# In[65]:


# ui_b_count_rank_in_uc
path_df = open(path_df_part_1_uic_label, 'r')
df_part_1_uic = pd.read_csv(path_df, index_col = False)
df_part_1_ui_b_count = pd.merge(df_part_1_uic, df_part_1_ui_b_count, on = ['user_id','item_id'], how = 'left')
df_part_1_ui_b_count['ui_b_count_rank_in_uc'] = df_part_1_ui_b_count.groupby(['user_id','item_category'])['ui_b_count_rank_in_u'].rank(method='min',ascending=True).astype('int')


# In[66]:


# ui_b_last_time
df_part_1.sort_values(by=['user_id','item_id','behavior_type','time'], inplace=True)
df_part_1_ui_b_last_time = df_part_1.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','time']]

df_part_1_ui_b_last_time['ui_b1_last_time'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['behavior_type'] == 1]['time']
df_part_1_ui_b_last_time['ui_b2_last_time'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['behavior_type'] == 2]['time']
df_part_1_ui_b_last_time['ui_b3_last_time'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['behavior_type'] == 3]['time']
df_part_1_ui_b_last_time['ui_b4_last_time'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['behavior_type'] == 4]['time']

df_part_1_ui_b_last_time.loc[df_part_1_ui_b_last_time['ui_b1_last_time'].notnull(), 'ui_b1_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_ui_b_last_time['ui_b1_last_time'])             
df_part_1_ui_b_last_time['ui_b1_last_hours'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['ui_b1_last_hours'].notnull()]['ui_b1_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_1_ui_b_last_time.loc[df_part_1_ui_b_last_time['ui_b2_last_time'].notnull(), 'ui_b2_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_ui_b_last_time['ui_b2_last_time'])             
df_part_1_ui_b_last_time['ui_b2_last_hours'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['ui_b2_last_hours'].notnull()]['ui_b2_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_1_ui_b_last_time.loc[df_part_1_ui_b_last_time['ui_b3_last_time'].notnull(), 'ui_b3_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_ui_b_last_time['ui_b3_last_time'])             
df_part_1_ui_b_last_time['ui_b3_last_hours'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['ui_b3_last_hours'].notnull()]['ui_b3_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_1_ui_b_last_time.loc[df_part_1_ui_b_last_time['ui_b4_last_time'].notnull(), 'ui_b4_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_ui_b_last_time['ui_b4_last_time'])             
df_part_1_ui_b_last_time['ui_b4_last_hours'] = df_part_1_ui_b_last_time[df_part_1_ui_b_last_time['ui_b4_last_hours'].notnull()]['ui_b4_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)


# In[67]:


df_part_1_ui_b_last_time = df_part_1_ui_b_last_time[['user_id',
                                                     'item_id',
                                                     'ui_b1_last_hours',
                                                     'ui_b2_last_hours',
                                                     'ui_b3_last_hours',
                                                     'ui_b4_last_hours']] 
df_part_1_ui_b_last_time = df_part_1_ui_b_last_time.groupby(['user_id', 'item_id']).agg({'ui_b1_last_hours': np.sum,
                                                                                         'ui_b2_last_hours': np.sum,
                                                                                         'ui_b3_last_hours': np.sum,
                                                                                         'ui_b4_last_hours': np.sum})
df_part_1_ui_b_last_time.reset_index(inplace = True)

# merge for generation of f_UI_part_1
f_UI_part_1 = pd.merge(df_part_1_ui_b_count, df_part_1_ui_b_last_time, how='left', on=['user_id', 'item_id'])

# write to csv file
f_UI_part_1.to_csv(path_df_part_1_UI, index = False)

f_UI_part_1.head()


# In[68]:


path_df = open(path_df_part_1, 'r')
df_part_1 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
df_part_1.columns = ['time','user_id','item_id','behavior_type','item_category']
# uc_b_count_in_6
df_part_1['cumcount'] = df_part_1.groupby(['user_id', 'item_category', 'behavior_type']).cumcount()
df_part_1_uc_b_count_in_6 = df_part_1.drop_duplicates(['user_id','item_category','behavior_type'],'last')[['user_id','item_category','behavior_type','cumcount']]
df_part_1_uc_b_count_in_6 = pd.get_dummies(df_part_1_uc_b_count_in_6['behavior_type']).join(df_part_1_uc_b_count_in_6[['user_id','item_category','cumcount']])
df_part_1_uc_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_1_uc_b_count_in_6['uc_b1_count_in_6'] = df_part_1_uc_b_count_in_6['behavior_type_1'] * (df_part_1_uc_b_count_in_6['cumcount']+1)
df_part_1_uc_b_count_in_6['uc_b2_count_in_6'] = df_part_1_uc_b_count_in_6['behavior_type_2'] * (df_part_1_uc_b_count_in_6['cumcount']+1)
df_part_1_uc_b_count_in_6['uc_b3_count_in_6'] = df_part_1_uc_b_count_in_6['behavior_type_3'] * (df_part_1_uc_b_count_in_6['cumcount']+1)
df_part_1_uc_b_count_in_6['uc_b4_count_in_6'] = df_part_1_uc_b_count_in_6['behavior_type_4'] * (df_part_1_uc_b_count_in_6['cumcount']+1)
df_part_1_uc_b_count_in_6 = df_part_1_uc_b_count_in_6[['user_id', 
                                                       'item_category', 
                                                       'uc_b1_count_in_6', 
                                                       'uc_b2_count_in_6', 
                                                       'uc_b3_count_in_6',
                                                       'uc_b4_count_in_6']]
df_part_1_uc_b_count_in_6 = df_part_1_uc_b_count_in_6.groupby(['user_id', 'item_category']).agg({'uc_b1_count_in_6': np.sum,
                                                                                                 'uc_b2_count_in_6': np.sum,
                                                                                                 'uc_b3_count_in_6': np.sum,
                                                                                                 'uc_b4_count_in_6': np.sum})
df_part_1_uc_b_count_in_6.reset_index(inplace = True)
df_part_1_uc_b_count_in_6['uc_b_count_in_6'] = df_part_1_uc_b_count_in_6['uc_b1_count_in_6'] +                                                df_part_1_uc_b_count_in_6['uc_b2_count_in_6'] +                                                df_part_1_uc_b_count_in_6['uc_b3_count_in_6'] +                                                df_part_1_uc_b_count_in_6['uc_b4_count_in_6']


# In[69]:


# uc_b_count_in_3
df_part_1_in_3 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-25')]
df_part_1_in_3['cumcount'] = df_part_1_in_3.groupby(['user_id', 'item_category', 'behavior_type']).cumcount()
df_part_1_uc_b_count_in_3 = df_part_1.drop_duplicates(['user_id','item_category','behavior_type'],'last')[['user_id','item_category','behavior_type','cumcount']]
df_part_1_uc_b_count_in_3 = pd.get_dummies(df_part_1_uc_b_count_in_3['behavior_type']).join(df_part_1_uc_b_count_in_3[['user_id','item_category','cumcount']])
df_part_1_uc_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_1_uc_b_count_in_3['uc_b1_count_in_3'] = df_part_1_uc_b_count_in_3['behavior_type_1'] * (df_part_1_uc_b_count_in_3['cumcount']+1)
df_part_1_uc_b_count_in_3['uc_b2_count_in_3'] = df_part_1_uc_b_count_in_3['behavior_type_2'] * (df_part_1_uc_b_count_in_3['cumcount']+1)
df_part_1_uc_b_count_in_3['uc_b3_count_in_3'] = df_part_1_uc_b_count_in_3['behavior_type_3'] * (df_part_1_uc_b_count_in_3['cumcount']+1)
df_part_1_uc_b_count_in_3['uc_b4_count_in_3'] = df_part_1_uc_b_count_in_3['behavior_type_4'] * (df_part_1_uc_b_count_in_3['cumcount']+1)
df_part_1_uc_b_count_in_3 = df_part_1_uc_b_count_in_3[['user_id', 
                                                       'item_category', 
                                                       'uc_b1_count_in_3', 
                                                       'uc_b2_count_in_3', 
                                                       'uc_b3_count_in_3',
                                                       'uc_b4_count_in_3']]
df_part_1_uc_b_count_in_3 = df_part_1_uc_b_count_in_3.groupby(['user_id', 'item_category']).agg({'uc_b1_count_in_3': np.sum,
                                                                                                 'uc_b2_count_in_3': np.sum,
                                                                                                 'uc_b3_count_in_3': np.sum,
                                                                                                 'uc_b4_count_in_3': np.sum})
df_part_1_uc_b_count_in_3.reset_index(inplace = True)
df_part_1_uc_b_count_in_3['uc_b_count_in_3'] = df_part_1_uc_b_count_in_3['uc_b1_count_in_3'] +                                                df_part_1_uc_b_count_in_3['uc_b2_count_in_3'] +                                                df_part_1_uc_b_count_in_3['uc_b3_count_in_3'] +                                                df_part_1_uc_b_count_in_3['uc_b4_count_in_3']


# In[70]:


# uc_b_count_in_1
df_part_1_in_1 = df_part_1[df_part_1['time'] >= np.datetime64('2014-11-27')]
df_part_1_in_1['cumcount'] = df_part_1_in_1.groupby(['user_id', 'item_category', 'behavior_type']).cumcount()
df_part_1_uc_b_count_in_1 = df_part_1_in_1.drop_duplicates(['user_id','item_category','behavior_type'], 'last')[['user_id','item_category','behavior_type','cumcount']]
df_part_1_uc_b_count_in_1 = pd.get_dummies(df_part_1_uc_b_count_in_1['behavior_type']).join(df_part_1_uc_b_count_in_1[['user_id','item_category','cumcount']])
df_part_1_uc_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)
df_part_1_uc_b_count_in_1['uc_b1_count_in_1'] = df_part_1_uc_b_count_in_1['behavior_type_1'] * (df_part_1_uc_b_count_in_1['cumcount']+1)
df_part_1_uc_b_count_in_1['uc_b2_count_in_1'] = df_part_1_uc_b_count_in_1['behavior_type_2'] * (df_part_1_uc_b_count_in_1['cumcount']+1)
df_part_1_uc_b_count_in_1['uc_b3_count_in_1'] = df_part_1_uc_b_count_in_1['behavior_type_3'] * (df_part_1_uc_b_count_in_1['cumcount']+1)
df_part_1_uc_b_count_in_1['uc_b4_count_in_1'] = df_part_1_uc_b_count_in_1['behavior_type_4'] * (df_part_1_uc_b_count_in_1['cumcount']+1)
df_part_1_uc_b_count_in_1 = df_part_1_uc_b_count_in_1[['user_id',
                                                       'item_category', 
                                                       'uc_b1_count_in_1', 
                                                       'uc_b2_count_in_1', 
                                                       'uc_b3_count_in_1',
                                                       'uc_b4_count_in_1']]
df_part_1_uc_b_count_in_1 = df_part_1_uc_b_count_in_1.groupby(['user_id', 'item_category']).agg({'uc_b1_count_in_1': np.sum,
                                                                                                 'uc_b2_count_in_1': np.sum,
                                                                                                 'uc_b3_count_in_1': np.sum,
                                                                                                 'uc_b4_count_in_1': np.sum})
df_part_1_uc_b_count_in_1.reset_index(inplace = True)
df_part_1_uc_b_count_in_1['uc_b_count_in_1'] = df_part_1_uc_b_count_in_1['uc_b1_count_in_1'] +                                                df_part_1_uc_b_count_in_1['uc_b2_count_in_1'] +                                                df_part_1_uc_b_count_in_1['uc_b3_count_in_1'] +                                                df_part_1_uc_b_count_in_1['uc_b4_count_in_1']
                                             
df_part_1_uc_b_count = pd.merge(df_part_1_uc_b_count_in_6, df_part_1_uc_b_count_in_3, on = ['user_id','item_category'], how = 'left').fillna(0)
df_part_1_uc_b_count = pd.merge(df_part_1_uc_b_count, df_part_1_uc_b_count_in_1, on = ['user_id','item_category'], how = 'left').fillna(0)
df_part_1_uc_b_count[['uc_b1_count_in_6',
                      'uc_b2_count_in_6',
                      'uc_b3_count_in_6',
                      'uc_b4_count_in_6',
                       'uc_b_count_in_6',
                      'uc_b1_count_in_3',
                      'uc_b2_count_in_3',
                      'uc_b3_count_in_3',
                      'uc_b4_count_in_3',
                       'uc_b_count_in_3',
                      'uc_b1_count_in_1',
                      'uc_b2_count_in_1',
                      'uc_b3_count_in_1',
                      'uc_b4_count_in_1',
                       'uc_b_count_in_1']] = df_part_1_uc_b_count[['uc_b1_count_in_6',
                                                                   'uc_b2_count_in_6',
                                                                   'uc_b3_count_in_6',
                                                                   'uc_b4_count_in_6',
                                                                    'uc_b_count_in_6',
                                                                   'uc_b1_count_in_3',
                                                                   'uc_b2_count_in_3',
                                                                   'uc_b3_count_in_3',
                                                                   'uc_b4_count_in_3',
                                                                    'uc_b_count_in_3',
                                                                   'uc_b1_count_in_1',
                                                                   'uc_b2_count_in_1',
                                                                   'uc_b3_count_in_1',
                                                                   'uc_b4_count_in_1',
                                                                    'uc_b_count_in_1']].astype(int)


# In[71]:


# uc_b_count_rank_in_u
df_part_1_uc_b_count['uc_b_count_rank_in_u'] = df_part_1_uc_b_count.groupby(['user_id'])['uc_b_count_in_6'].rank(method='min',ascending=False).astype('int')

# uc_b_last_time
df_part_1.sort_values(by=['user_id','item_category','behavior_type','time'], inplace=True)
df_part_1_uc_b_last_time = df_part_1.drop_duplicates(['user_id','item_category','behavior_type'],'last')[['user_id','item_category','behavior_type','time']]

df_part_1_uc_b_last_time['uc_b1_last_time'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['behavior_type'] == 1]['time']
df_part_1_uc_b_last_time['uc_b2_last_time'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['behavior_type'] == 2]['time']
df_part_1_uc_b_last_time['uc_b3_last_time'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['behavior_type'] == 3]['time']
df_part_1_uc_b_last_time['uc_b4_last_time'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['behavior_type'] == 4]['time']

df_part_1_uc_b_last_time.loc[df_part_1_uc_b_last_time['uc_b1_last_time'].notnull(), 'uc_b1_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_uc_b_last_time['uc_b1_last_time'])             
df_part_1_uc_b_last_time['uc_b1_last_hours'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['uc_b1_last_hours'].notnull()]['uc_b1_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_1_uc_b_last_time.loc[df_part_1_uc_b_last_time['uc_b2_last_time'].notnull(), 'uc_b2_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_uc_b_last_time['uc_b2_last_time'])             
df_part_1_uc_b_last_time['uc_b2_last_hours'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['uc_b2_last_hours'].notnull()]['uc_b2_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_1_uc_b_last_time.loc[df_part_1_uc_b_last_time['uc_b3_last_time'].notnull(), 'uc_b3_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_uc_b_last_time['uc_b3_last_time'])             
df_part_1_uc_b_last_time['uc_b3_last_hours'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['uc_b3_last_hours'].notnull()]['uc_b3_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_1_uc_b_last_time.loc[df_part_1_uc_b_last_time['uc_b4_last_time'].notnull(), 'uc_b4_last_hours'] = (pd.to_datetime('2014-11-28') - df_part_1_uc_b_last_time['uc_b4_last_time'])             
df_part_1_uc_b_last_time['uc_b4_last_hours'] = df_part_1_uc_b_last_time[df_part_1_uc_b_last_time['uc_b4_last_hours'].notnull()]['uc_b4_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_1_uc_b_last_time = df_part_1_uc_b_last_time[['user_id',
                                                     'item_category',
                                                     'uc_b1_last_hours',
                                                     'uc_b2_last_hours',
                                                     'uc_b3_last_hours',
                                                     'uc_b4_last_hours']] 

df_part_1_uc_b_last_time = df_part_1_uc_b_last_time.groupby(['user_id', 'item_category']).agg({'uc_b1_last_hours': np.sum,
                                                                                               'uc_b2_last_hours': np.sum,
                                                                                               'uc_b3_last_hours': np.sum,
                                                                                               'uc_b4_last_hours': np.sum})
df_part_1_uc_b_last_time.reset_index(inplace = True)

# merge for generation of f_UC_part_1
f_UC_part_1 = pd.merge(df_part_1_uc_b_count, df_part_1_uc_b_last_time, how='left', on=['user_id', 'item_category'])

# write to csv file
f_UC_part_1.to_csv(path_df_part_1_UC, index = False)

