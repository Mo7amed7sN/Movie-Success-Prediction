import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from Preprocessing_Helper import Encode_Json
from Preprocessing_Helper import Label_Encode
from Preprocessing_Helper import Encode_Categorical
from Preprocessing_Helper import FeatureScaling

credits_data = pd.read_csv('tmdb_5000_credits.csv')
new_cls_feat = list()
new_reg_feat = list()


def regression_preprocessing():
    # read data from 2 files and concat..
    global credits_data
    Data = pd.read_csv('tmdb_5000_movies_train.csv')
    Data = pd.concat([Data, credits_data], axis=1)
    # print(Data.shape)

    # drop unneeded columns.
    cols = ['homepage', 'id', 'original_title', 'overview', 'release_date', 'status',
            'tagline', 'title', 'movie_id', 'keywords', 'production_companies', 'production_countries', 'spoken_languages', 'crew', 'cast']
    for i in cols:
        Data.drop([i], axis=1, inplace=True)
    # print(Data.shape)

    # fill 0 budget, rev, runtime with avg of col.
    avg_budget = Data['budget'].mean()
    avg_rev = Data['revenue'].mean()
    avg_run = Data['runtime'].mean()
    for i in range(len(Data.iloc[:, 0].values)):
        if Data.iloc[:, 0].values[i] == 0 or np.isnan(Data.iloc[:, 0].values[i]):  # check if nan or 0
            Data.iloc[:, 0].values[i] = avg_budget
    for i in range(len(Data.iloc[:, 4].values)):
        if Data.iloc[:, 4].values[i] == 0 or np.isnan(Data.iloc[:, 4].values[i]):  # check if nan or 0
            Data.iloc[:, 4].values[i] = avg_rev
    for i in range(len(Data.iloc[:, 5].values)):
        if Data.iloc[:, 5].values[i] == 0 or np.isnan(Data.iloc[:, 5].values[i]):  # check if nan or 0
            Data.iloc[:, 5].values[i] = avg_run

    # fill nan popularity and vote_count.
    avg_pop = Data['popularity'].mean()
    avg_vote = Data['vote_count'].mean()
    for i in range(len(Data.iloc[:, 3].values)):
        if np.isnan(Data.iloc[:, 3].values[i]):
            Data.iloc[:, 3].values[i] = avg_pop
    for i in range(len(Data.iloc[:, 7].values)):
        if np.isnan(Data.iloc[:, 7].values[i]):
            Data.iloc[:, 7].values[i] = avg_vote

    # Feature Scaling for numerical Data.
    Data = FeatureScaling(Data, [0, 3, 4, 5, 7])

    # Encode Categorical Col with one_hot_vector
    Data, feat = Encode_Categorical(Data, 'original_language')
    new_reg_feat.append(['original_language', feat])

    # Encode Json objects by 1_hot_vector.
    Data, feat = Encode_Json(Data, 'genres', 'name')
    new_reg_feat.append(['genres', feat])

    # Only Drop if Sample labeled with nan.
    Data.dropna(how='any', inplace=True)

    Y = Data['vote_average']  # output
    Data.drop(['vote_average'], axis=1, inplace=True)
    X = Data  # inputs

    # Dimensionality reduction
    '''pca = PCA(n_components=10)
    X = pca.fit_transform(X)'''

    # save the final data that will be used in training, validation, testing.
    X = pd.DataFrame(X)
    X.to_csv(r'Encoded_X_regression.csv')
    Y = pd.DataFrame(Y)
    Y.to_csv(r'Encoded_y_regression.csv')
    np.save('train2test_feat_regression.npy', np.array(new_reg_feat))


def classification_preprocessing():
    # read data from 2 files and concat..
    global credits_data
    Data = pd.read_csv('tmdb_5000_movies_classification.csv')
    Data = pd.concat([Data, credits_data], axis=1)
    # print(Data.shape)

    # drop unneeded columns.
    cols = ['homepage', 'id', 'original_title', 'overview', 'release_date', 'status',
            'tagline', 'title', 'movie_id', 'keywords', 'production_companies', 'production_countries',
            'crew', 'cast']

    for i in cols:
        Data.drop([i], axis=1, inplace=True)
    # print(Data.shape)

    # fill 0 budget, rev, runtime with avg of col.
    avg_budget = Data['budget'].mean()
    avg_rev = Data['revenue'].mean()
    avg_run = Data['runtime'].mean()
    for i in range(len(Data.iloc[:, 0].values)):
        if Data.iloc[:, 0].values[i] == 0 or np.isnan(Data.iloc[:, 0].values[i]):  # check if nan or 0
            Data.iloc[:, 0].values[i] = avg_budget
    for i in range(len(Data.iloc[:, 4].values)):
        if Data.iloc[:, 4].values[i] == 0 or np.isnan(Data.iloc[:, 4].values[i]):  # check if nan or 0
            Data.iloc[:, 4].values[i] = avg_rev
    for i in range(len(Data.iloc[:, 5].values)):
        if Data.iloc[:, 5].values[i] == 0 or np.isnan(Data.iloc[:, 5].values[i]):  # check if nan or 0
            Data.iloc[:, 5].values[i] = avg_run

    # fill nan popularity and vote_count.
    avg_pop = Data['popularity'].mean()
    avg_vote = Data['vote_count'].mean()
    for i in range(len(Data.iloc[:, 3].values)):
        if np.isnan(Data.iloc[:, 3].values[i]):
            Data.iloc[:, 3].values[i] = avg_pop
    for i in range(len(Data.iloc[:, 7].values)):
        if np.isnan(Data.iloc[:, 7].values[i]):
            Data.iloc[:, 7].values[i] = avg_vote

    # Label Encode Rate Col (Y) .
    Data = Label_Encode(Data, 'rate')
    Y = Data['rate']  # output
    Data.drop(['rate'], axis=1, inplace=True)
    X = Data  # inputs

    # Feature Scaling on numerical cols.
    X = FeatureScaling(X, [0, 3, 4, 5, 7])

    # Encode Cat.. col by 1_hot_vector too.
    X, feat = Encode_Categorical(X, 'original_language')
    new_cls_feat.append(['original_language', feat])

    # Encode Json objects by 1_hot_vector.
    X, feat = Encode_Json(X, 'genres', 'name')
    new_cls_feat.append(['genres', feat])

    X, feat = Encode_Json(X, 'spoken_languages', 'iso_639_1')
    new_cls_feat.append(['spoken_languages', feat])

    # Dimensionality reduction
    '''pca = PCA(n_components=100)
    X = pca.fit_transform(X)'''

    # save the final data that will be used in training, validation, testing.
    X = pd.DataFrame(X)
    X.to_csv(r'Encoded_X_classification.csv')
    Y = pd.DataFrame(Y)
    Y.to_csv(r'Encoded_y_classification.csv')
    np.save('train2test_feat_classification.npy', np.array(new_cls_feat))


# 149
classification_preprocessing()
# 48
regression_preprocessing()
