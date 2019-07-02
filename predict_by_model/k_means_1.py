"""""
dealing with positive and negative sample imbalance by k-means

"""""

### input

path_df_part_1_uic_label = "mobile/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "mobile/df_part_2_uic_label.csv"
path_df_part_3_uic       = "mobile/df_part_3_uic.csv"

path_df_part_1_U   = "mobile/feature/df_part_1_U.csv"  
path_df_part_1_I   = "mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "mobile/feature/df_part_1_UC.csv"

path_df_part_2_U   = "mobile/feature/df_part_2_U.csv"  
path_df_part_2_I   = "feature/df_part_2_I.csv"
path_df_part_2_C   = "feature/df_part_2_C.csv"
path_df_part_2_IC  = "feature/df_part_2_IC.csv"
path_df_part_2_UI  = "feature/df_part_2_UI.csv"
path_df_part_2_UC  = "feature/df_part_2_UC.csv"

path_df_part_3_U   = "feature/df_part_3_U.csv"  
path_df_part_3_I   = "feature/df_part_3_I.csv"
path_df_part_3_C   = "feature/df_part_3_C.csv"
path_df_part_3_IC  = "feature/df_part_3_IC.csv"
path_df_part_3_UI  = "feature/df_part_3_UI.csv"
path_df_part_3_UC  = "feature/df_part_3_UC.csv"

### out file

path_df_part_1_uic_label_0 = "mobile/gbdt/k_means_subsample/df_part_1_uic_label_0.csv"
path_df_part_1_uic_label_1 = "mobile/gbdt/k_means_subsample/df_part_1_uic_label_1.csv"


path_df_part_1_uic_label_cluster = "mobile/gbdt/k_means_subsample/df_part_1_uic_label_cluster.csv"

path_df_part_1_scaler = "mobile/gbdt/k_means_subsample/df_part_1_scaler"




import pandas as pd
import numpy as np

def df_read(path, mode = 'r'):
    path_df = open(path, mode)
    df = pd.read_csv(path_df, index_col = False)
    return   df

def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size
    
    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df) : 
        return df
    else : 
        return df.sample(n = sub_size)  



df_part_1_uic_label = df_read(path_df_part_1_uic_label)
df_part_1_uic_label_0 = df_part_1_uic_label[df_part_1_uic_label['label'] == 0]
df_part_1_uic_label_1 = df_part_1_uic_label[df_part_1_uic_label['label'] == 1]
df_part_1_uic_label_0.to_csv(path_df_part_1_uic_label_0, index=False)
df_part_1_uic_label_1.to_csv(path_df_part_1_uic_label_1, index=False)


# clustering based on sklearn
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import pickle

##### part_1 #####

df_part_1_U  = df_read(path_df_part_1_U )   
df_part_1_I  = df_read(path_df_part_1_I )
df_part_1_C  = df_read(path_df_part_1_C )
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

scaler_1 = preprocessing.StandardScaler() 
batch = 0
for df_part_1_uic_label_0 in pd.read_csv(open(path_df_part_1_uic_label_0, 'r'), chunksize=150000): 
    # construct of part_1's sub-training set
    train_data_df_part_1 = pd.merge(df_part_1_uic_label_0, df_part_1_U, how='left', on=['user_id'])
    train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_I,  how='left', on=['item_id'])
    train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_C,  how='left', on=['item_category'])
    train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    # getting all the complete features for clustering
    train_X_1 = train_data_df_part_1.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                                    'u_b4_rate',
                                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                                    'i_b4_rate',
                                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                                    'c_b4_rate',
                                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                                    'uc_b_count_rank_in_u'])
    # feature standardization
    scaler_1.partial_fit(train_X_1)        
        
    batch += 1
    print('chunk %d done.' %batch) 


# initial clusters
mbk_1 = MiniBatchKMeans(init='k-means++', n_clusters=1000, batch_size=500, reassignment_ratio=10**-4) 
classes_1 = []
batch = 0
for df_part_1_uic_label_0 in pd.read_csv(open(path_df_part_1_uic_label_0, 'r'), chunksize=15000): 
 
        # construct of part_1's sub-training set
        train_data_df_part_1 = pd.merge(df_part_1_uic_label_0, df_part_1_U, how='left', on=['user_id'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_I,  how='left', on=['item_id'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_C,  how='left', on=['item_category'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_IC, how='left', on=['item_id','item_category'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UC, how='left', on=['user_id','item_category'])
        
        train_X_1 = train_data_df_part_1.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                                    'u_b4_rate',
                                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                                    'i_b4_rate',
                                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                                    'c_b4_rate',
                                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                                    'uc_b_count_rank_in_u'])
        # feature standardization
        standardized_train_X_1 = scaler_1.transform(train_X_1)
         
        # fit clustering model
        mbk_1.partial_fit(standardized_train_X_1)
    
        print(len(np.unique(mbk_1.labels_)))
        
     
        classes_1 = np.append(classes_1, mbk_1.labels_)
        
        batch += 1
        print('chunk %d done.' %batch) 

pickle.dump(scaler_1, open(path_df_part_1_scaler,'wb')) 

# add a new attr for keys
df_part_1_uic_label_0 = df_read(path_df_part_1_uic_label_0)
df_part_1_uic_label_1 = df_read(path_df_part_1_uic_label_1)
df_part_1_uic_label_0.head()

df_part_1_uic_label_0['class'] = classes_1.astype('int') + 1
df_part_1_uic_label_1['class'] = 0
df_part_1_uic_label_0.head()


# add a new attr for keys
df_part_1_uic_label_0 = df_read(path_df_part_1_uic_label_0)
df_part_1_uic_label_1 = df_read(path_df_part_1_uic_label_1)

df_part_1_uic_label_0['class'] = classes_1.astype('int') + 1
df_part_1_uic_label_1['class'] = 0

df_part_1_uic_label_class = pd.concat([df_part_1_uic_label_0, df_part_1_uic_label_1])
df_part_1_uic_label_class.to_csv(path_df_part_1_uic_label_cluster, index=False)
