
# coding: utf-8

# In[212]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[ ]:





# In[214]:



def drop_nan(df, axis = 0):
    if axis == 0:
        return df.dropna(axis = 0)
    elif axis == 1:
        return df.dropna(axis = 1)
    else:
        raise Exception('wrong axis')
                    


def replace(df, columns, stat = 'mean'):
    for column in columns:
        new_column = pd.DataFrame(df[column])
        if stat == 'mean':
            new_column = new_column.apply(lambda x: x.fillna(x.mean()))
        elif stat == 'mode':
            new_column = new_column.apply(lambda x: x.fillna(x.mode()))
        elif stat == 'median':
            new_column = new_column.apply(lambda x: x.fillna(x.median()))
        else:
            raise Exception('wrong stat')
        df[column] = new_column
    return df


def lr_replace(df, target, columns):
    linear = LinearRegression()
    columns_all_data = [i for i in columns]
    columns_all_data.append(target)
    all_data = pd.DataFrame(df[columns_all_data])
    not_nans = drop_nan(all_data)
    X_train = not_nans.drop(target, axis = 1)
    y_train = not_nans[[target]]
    linear.fit(X_train, y_train)
    not_nan_x = all_data.drop(target, axis = 1)
    all_data_new = replace(all_data, all_data[columns], 'mean')
    all_data_new = all_data_new.fillna('clear')
    indexes = []

    for i in range(len(all_data_new[target])):
        if all_data_new[target][i] == 'clear':
            indexes.append(i)
    all_data_new.loc[indexes,target] = linear.predict(all_data_new.iloc[indexes,].loc[:,columns]) 
    all_data_new.loc[indexes,target] = linear.predict(all_data_new.iloc[indexes,].loc[:,columns]) 
    return all_data_new


def standardize(df, label):
    df = df.copy(deep=True)
    series = df.loc[:, label]
    avg = series.mean()
    stdv = series.std()
    series_standardized = (series - avg)/ stdv
    df[label] = series_standardized
    return df

def normalize(df, label):
    df = df.copy(deep=True)
    series = df.loc[:, label]
    min_ = series.min()
    max_ = series.max()
    print(series)
    series_normalized = (series - min_)/(max_ - min_) 
    df[label] = series_normalized
    return df

def fill_knn(df, target, k=2):
    train = df[df[target].notnull()]
    test = df[df[target].isnull()]
    list_of_nans = df[df[target].isnull() == True].index.tolist()
    column_index = df.columns.get_loc(target)
    for row in range(len(test)): 
        distances_list = []
        for i in range(len(train)):
            distance = 0 
            for j in range(len(train.iloc[i])):
                if train.iloc[i,j] != test.iloc[row,j]:
                    distance += 1  
            distances_list.append((distance,i))
        list_of_nearest = [new[1] for new in sorted(distances_list, key=lambda distance: distance[0])]
        predicted_one = df.loc[list_of_nearest[:k],target].value_counts().idxmax()
        test.iloc[row,column_index] = predicted_one
    result = pd.concat([test,train])
    return result

