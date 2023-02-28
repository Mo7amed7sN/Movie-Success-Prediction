import numpy as np
import pandas as pd
from ast import literal_eval


def Encode_Json(dataset, choice, IDN):  # Dataset name , choice col name (ex'genres'), IDN the key of json object 'name'
    dataset[choice] = dataset[choice].fillna('[]').apply(literal_eval).apply(lambda x: [ind[IDN] for ind in x] if isinstance(x, list) else [])
    new_features = np.array(dataset[choice])

    freq = dict()

    unique_features = set()
    for i in new_features:
        for j in i:
            if str(j) == 'nan':
                continue
            unique_features.add(j)
            try:
                freq[j] = freq[j] + 1
            except:
                freq[j] = 1

    sorted_tuple = sorted(freq.items(), reverse=True, key=lambda x: x[1])
    if len(unique_features) > 100:
        unique_features.clear()
        cnt = 0
        for elem in sorted_tuple:
            if cnt == 50:
                break
            unique_features.add(elem[0])
            cnt = cnt + 1

    num = np.zeros((dataset.shape[0], len(unique_features)))
    ret_Data = pd.DataFrame(num, columns=unique_features)  # the new dataframe table.
    for index, i in enumerate(new_features):
        for j in i:
            if str(j) == 'nan':
                continue
            try:
                ret_Data.iloc[index].loc[j] = 1
            except:
                continue

    dataset.drop([choice], axis=1, inplace=True)
    ret_Data = pd.concat([dataset, ret_Data], axis=1)
    return ret_Data, unique_features


def Label_Encode(X, col):
    for i in range(X.shape[0]):
        if X.loc[i, col] == 'Low':
            X.loc[i, col] = 0
        elif X.loc[i, col] == 'Intermediate':
            X.loc[i, col] = 1
        elif X.loc[i, col] == 'High':
            X.loc[i, col] = 2
    return X


def Encode_Categorical(X, choice):
    vals = X[choice].values

    uinque_lang = set()
    for i in vals:
        if str(i) == 'nan':
            continue
        uinque_lang.add(i)

    num = np.zeros((X.shape[0], len(uinque_lang)))
    ret_Data = pd.DataFrame(num, columns=uinque_lang)  # the new dataframe table.

    for i in range(X.shape[0]):
        val = X.loc[i, 'original_language']
        if str(val) == 'nan':
            continue
        ret_Data.loc[i, val] = 1

    X.drop([choice], axis=1, inplace=True)
    ret_Data = pd.concat([X, ret_Data], axis=1)
    return ret_Data, uinque_lang


def FeatureScaling(X, cols):
    NX = X
    for i in cols:
        NX.iloc[:, i] = (NX.iloc[:, i] - np.min(NX.iloc[:, i])) / (
                np.max(NX.iloc[:, i]) - np.min(NX.iloc[:, i]))
    return NX


def testing_Encoder(Data, identifier):
    NDATA = Data

    # New features of Reg, Cls Data.
    new_feat = None
    if identifier == 'r':
        new_feat = np.load('train2test_feat_regression.npy', allow_pickle=True)
    else:
        new_feat = np.load('train2test_feat_classification.npy', allow_pickle=True)

    # map each col with it,s new features.
    unique_features = dict()
    for i in new_feat:
        unique_features[i[0]] = i[1]

    # encode original_language col.
    num = np.zeros((NDATA.shape[0], len(unique_features['original_language'])))
    ret_Data = pd.DataFrame(num, columns=unique_features['original_language'])  # the new dataframe table.
    for i in range(NDATA.shape[0]):
        val = NDATA.loc[i, 'original_language']
        if val in unique_features['original_language']:
            ret_Data.loc[i, val] = 1
    NDATA.drop(['original_language'], axis=1, inplace=True)
    NDATA = pd.concat([NDATA, ret_Data], axis=1)

    # encode other Json cols.
    cols = ['genres']
    if identifier == 'c':
        cols.append('spoken_languages')
    IDN = None
    for i in cols:
        if i == 'spoken_languages':
            IDN = 'iso_639_1'
        elif i == 'production_countries':
            IDN = 'iso_3166_1'
        else:
            IDN = 'name'

        NDATA[i] = NDATA[i].fillna('[]').apply(literal_eval).apply(
            lambda x: [indd[IDN] for indd in x] if isinstance(x, list) else [])
        new_features = np.array(NDATA[i])
        num = np.zeros((NDATA.shape[0], len(unique_features[i])))
        ret_Data = pd.DataFrame(num, columns=unique_features[i])  # the new dataframe table.

        for ind in range(NDATA.shape[0]):
            for new in unique_features[i]:
                if new in new_features[ind]:
                    ret_Data.loc[ind, new] = 1
        NDATA.drop([i], axis=1, inplace=True)
        NDATA = pd.concat([NDATA, ret_Data], axis=1)
    return NDATA
