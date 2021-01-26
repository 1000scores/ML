# To add a new cell, type ' '
# To add a new markdown cell, type '  [markdown]'

from IPython import get_ipython


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import lognorm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def preprocess(path='train-2.csv'):
    df = pd.read_csv(path, converters={'features': eval})
    df = df.rename(str.lower, axis='columns',)
    df = df.set_index('id')



    y = df['target']
    X = df.drop(['target'], axis=1)

    high_percent = len(df[df['target'] == 'high']) * 100 / len(df)
    medium_percent = len(df[df['target'] == 'medium']) * 100 / len(df)
    low_percent = len(df[df['target'] == 'low']) * 100 / len(df)


    df['temp_target'] = df['target'].map({'low' : low_percent, 'medium' : medium_percent + low_percent, 'high' : 1})



    df['most_frequent_house_comment'] = df.groupby('building_id')['temp_target'].transform(lambda x: x.sum() / len(x))
    # Значит многие одинаковые здания имеют одинаковые отзывы



    df['most_frequent_rieltor_comment'] = df.groupby('manager_id')['temp_target'].transform(lambda x: x.sum() / len(x))



    df['description'] = df['description'].str.lower()



    def remove_trash(cur_str):
        cur_str = str(cur_str)
        nw_str = ""
        for elem in cur_str:
            if (elem >= 'a' and elem <= 'z') or elem == ' ':
                nw_str += elem

        return nw_str.split(' ')

    df['description'] = df['description'].apply(remove_trash)



    description_words_weight = dict()
    description_words_count = dict()
    for index, row in df.iterrows():
        for word in row.description:
            if len(word) < 3:
                continue
            if word in description_words_weight:
                description_words_weight[word] += row.temp_target
                description_words_count[word] += 1
            else:
                description_words_weight[word] = row.temp_target
                description_words_count[word] = 1

    for key, value in description_words_weight.items():
        description_words_weight[key] = value / (description_words_count[key] * 100)




    def words_weight(x):
        res = 0.0
        for word in x:
            if word in description_words_weight:
                res += description_words_weight[word]
        if len(x) > 0:
            res /= len(x)
        return res

    df['description'] = df['description'].apply(words_weight)





    features_words_weight = dict()
    features_words_count = dict()
    for index, row in df.iterrows():
        for word in row.features:
            word = str.lower(word)
            if word in features_words_weight:
                features_words_weight[word] += row.temp_target
                features_words_count[word] += 1
            else:
                features_words_weight[word] = row.temp_target
                features_words_count[word] = 1


    for key, value in features_words_weight.items():
        features_words_weight[key] = value / (features_words_count[key] * 100)


    def features_weight(x):
        res = 0.0
        for word in x:
            word = str.lower(word)
            if word in features_words_weight:
                res += features_words_weight[word]
        if len(x) > 0:
            res /= len(x)
        return res

    #display(features_weight(df.iloc[0, 6]))
    df['features'] = df['features'].apply(features_weight)


    df.drop(['building_id', 'created', 'display_address', 'latitude', 'listing_id', 'longitude', 'manager_id', 'photos', 'street_address'], inplace = True, axis = 1)



    df.drop(['temp_target'], inplace = True, axis = 1)


    y = df['target'].map({'low' : 1, 'medium' : 2, 'high' : 3})
    X = df.drop(['target'], axis = 1)



    standart_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    X[X.columns] = minmax_scaler.fit_transform(X[X.columns])

    return X, y

