import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from Preprocessing_Helper import FeatureScaling
from Preprocessing_Helper import testing_Encoder
from Preprocessing_Helper import Label_Encode

credits_data = pd.read_csv('tmdb_5000_credits_test1.csv')


def regression_testing_preprocessing():
    # read data from 2 files and concat..
    global credits_data
    Data = pd.read_csv('tmdb_5000_movies_testing_regression1.csv')
    # Data = pd.concat([Data, credits_data], axis=1)
    # print(Data.shape)

    # drop unneeded columns.
    cols = ['homepage', 'id', 'original_title', 'overview', 'release_date', 'status',
            'tagline', 'title',  'keywords', 'production_companies', 'production_countries',
            'spoken_languages']
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

    # Feature Scale to numerical Data.
    Data = FeatureScaling(Data, [0, 3, 4, 5, 7])

    # Encode all text Data with features that used in training [represented as one_hot_vector].
    Data = testing_Encoder(Data, 'r')

    # Only Drop if Sample labeled with nan.
    Data.dropna(how='any', inplace=True)

    Y = Data['vote_average']  # output
    Data.drop(['vote_average'], axis=1, inplace=True)
    X = Data  # inputs

    # Dimensionality reduction
    '''pca = PCA(n_components=9)
    X = pca.fit_transform(X)'''

    # save the final data that will be used in training, validation, testing.
    X = pd.DataFrame(X)
    X.to_csv(r'Encoded_X_testing_regression.csv')
    Y = pd.DataFrame(Y)
    Y.to_csv(r'Encoded_y_testing_regression.csv')


def classification_testing_preprocessing():
    # read data from 2 files and concat..
    global credits_data
    Data = pd.read_excel('tmdb_5000_movies_testing_classification1.xlsx')
    # Data = pd.concat([Data, credits_data], axis=1)
    # print(Data.shape)

    # drop unneeded columns.
    cols = ['homepage', 'id', 'original_title', 'overview', 'release_date', 'status',
            'tagline', 'title', 'keywords', 'production_companies', 'production_countries']
    for i in cols:
        Data.drop([i], axis=1, inplace=True)
    # print(Data.shape)

    # fill 0 budget, rev, runtime with avg of col
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

    # feature scale numerical Data.
    Data = FeatureScaling(Data, [0, 3, 4, 5, 7])
    # Encode all text Data with features that used in training [represented as one_hot_vector].
    Data = testing_Encoder(Data, 'c')
    # Label Encode Rate Col.
    Data = Label_Encode(Data, 'rate')

    # Only Drop if Sample labeled with nan.
    Data.dropna(how='any', inplace=True)

    Y = Data['rate']  # output
    Data.drop(['rate'], axis=1, inplace=True)
    X = Data  # inputs

    # Dimensionality reduction
    '''pca = PCA(n_components=9)
    X = pca.fit_transform(X)'''

    # save the final data that will be used in training, validation, testing.
    X = pd.DataFrame(X)
    X.to_csv(r'Encoded_X_testing_classification.csv')
    Y = pd.DataFrame(Y)
    Y.to_csv(r'Encoded_y_testing_classification.csv')


regression_testing_preprocessing()
classification_testing_preprocessing()
