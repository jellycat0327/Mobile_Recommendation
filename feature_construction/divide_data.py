"""

Due to the effect of user's behaviour a week ago on the purchase on the test date is small, we only select one week of data for predicting. 

Dividing the data into three parts:

part1: train: 11.22 ~ 11.27 tar = 11.28
part2: train: 11.29 ~ 12.04 tar = 12.05
part3: test: 12.13 ~ 12.18

"""

import pandas as pd

### input
path_df_D = "tianchi_fresh_comp_train_user.csv"

### output
path_df_part_1 = "mobile/df_part_1.csv"
path_df_part_2 = "mobile/df_part_2.csv"
path_df_part_3 = "mobile/df_part_3.csv"

path_df_part_1_tar = "mobile/df_part_1_tar.csv"
path_df_part_2_tar = "mobile/df_part_2_tar.csv"

path_df_part_1_uic_label = "mobile/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "mobile/df_part_2_uic_label.csv"
path_df_part_3_uic       = "mobile/df_part_3_uic.csv"

batch = 0
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
for df in pd.read_csv(open(path_df_D, 'r'), parse_dates=['time'], index_col = ['time'], date_parser = dateparse, chunksize = 100000):
    
    df_part_1 = df['2014-11-22':'2014-11-27']
    df_part_1_tar = df['2014-11-28']
    df_part_2 = df['2014-11-29':'2014-12-04']
    df_part_2_tar = df['2014-12-05']
    df_part_3 = df['2014-12-13':'2014-12-18']
    
    df_part_1.to_csv(path_df_part_1,  
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a') 
    
    df_part_1_tar.to_csv(path_df_part_1_tar,
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')
    
    df_part_2.to_csv(path_df_part_2,  
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a') 
    
    df_part_2_tar.to_csv(path_df_part_2_tar,
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')  
    
    df_part_3.to_csv(path_df_part_3,  
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')
    batch += 1
    print('chunk %d done.' %batch) 
    







