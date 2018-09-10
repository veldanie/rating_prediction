# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:30:41 2018

@author: daniel.velasquez
"""
def is_string(s):
    try:
        float(s)
        return False
    except ValueError:
        return True


def model_training(data, feat_key, le, remove_nan, perc_train_size, output_file, model_file, sov_encoder_file, n_estimators = 500, min_samples_leaf = 1):

    #import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import check_random_state
    from sklearn.externals import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import tree

    data_index = data.index # Se crea la variable data_index para publicar el output.
    y_ = np.array(data.pop('IssuerRating'))
    X_ = np.array(data[feat_key["Key"]])


    # Remove observations with no output
    ind_valid_out = [is_string(yi) for yi in y_]
    X = X_[ind_valid_out]
    y = y_[ind_valid_out]
    data_index = data_index[ind_valid_out]
    # Encode y values,
    y = np.array([list(le.loc[yi])[0] if is_string(yi) else float('NaN') for yi in y])


    # Encode Sovereig rating
    sr = feat_key[feat_key["Key"] == 'SovereignRating']
    if len(sr)>0:
        pos_sr = feat_key.index.get_loc(sr.index[0])# Position sovereign rating
        pos_str = [is_string(x) for x in X[:,pos_sr]]
        labels = np.unique(X[pos_str,pos_sr])
        le_X = LabelEncoder()
        le_X.fit(labels)
        X[pos_str,pos_sr] = le_X.transform(X[pos_str,pos_sr])
        joblib.dump(le_X, sov_encoder_file)# Save sovereign label encoder

    # Remove NaN
    if remove_nan:
        ind_not_na = [not np.isnan(np.sum(x)) for x in X]
        X = X[ind_not_na]
        y = y[ind_not_na]
        data_index = data_index[ind_not_na]
    else:
        imp = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
        imp.fit(X = X_train)
        X = imp.transform(X)


    # Data Permitation:
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])

    X = X[permutation]
    y = y[permutation]
    data_index = data_index[permutation]

    # Train and test samples:

    train_size = int(X.shape[0] * perc_train_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, shuffle = False)


    print('Muestra de entrenamiento: %d' % X_train.shape[0])
    print('Muestra de testing: %d' % X_test.shape[0])
    print('')

    # Model fitting:
    clf = RandomForestClassifier(n_estimators = n_estimators, max_features = "auto", min_samples_leaf = min_samples_leaf)
    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, model_file)

    score = clf.score(X_test, y_test)
    print('Score sobre muestra de testing:')
    print(score)
    print('')


    # output file:

    pred_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in clf.predict(X_test)])
    y_test_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in y_test])

    if len(sr)>0:
        X_test[:, pos_sr] = le_X.inverse_transform(X_test[:, pos_sr].astype('int')) # Inverse transform of sov. ratingsS

    data_test = pd.DataFrame(np.column_stack((np.column_stack((X_test, y_test_calif)), pred_calif)), columns = list(feat_key.index)+['Rating Test', 'Rating Predicc'], index = data_index[np.arange(train_size, data_index.shape[0])])

    # Output file:
    data_test.to_csv(output_file)

    # Variables importances:
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print('')
    print("Ranking:")
    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, feat_key.index[indices[f]], importances[indices[f]]))

    # Plot importances:
    print('')
    plt.figure()
    plt.title("Importancias")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")

    plt.xticks(range(X.shape[1]), np.arange(X.shape[1])+1)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    return(None)


def rating_prediction(data, model1, model2, feat_key, le, sov_lab_encoder, output_file):
    # model1: model inicial que considera la calif. soberana.
    # model2: model que no tiene en cuenta calif. soberana.
    import numpy as np
    import pandas as pd

    X_new = np.array(data.loc[feat_key.index].T)
    X_new_pure = np.array(data.loc[feat_key.index[(feat_key != 'SovereignRating')['Key']]].T)

    if sov_lab_encoder != None:
        pos_sr = feat_key.index.get_loc(feat_key[feat_key["Key"] == 'SovereignRating'].index[0])# Position sovereign rating
        sob_rating = X_new[:,pos_sr].copy()
        X_new[:,pos_sr] = sov_lab_encoder.transform(X_new[:,pos_sr])

    pred_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in model1.predict(X_new)])

    X_new[:, pos_sr] = sov_lab_encoder.inverse_transform(X_new[:, pos_sr].astype('int')) # Inverse transform of sov. ratingsS

    pred_calif_pure = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in model2.predict(X_new_pure)])

    rat_dist = np.array([int(le.loc[x]) for x in sob_rating]) - np.array([int(le.loc[x]) for x in pred_calif_pure])
    rat_trans = int(le.loc['AAA'])  - np.array([np.max([int(i),0]) for i in rat_dist])
    pred_calif_translate = [le[le['Value']==x].index[0] for x in rat_trans]

    data_pred = pd.DataFrame(np.column_stack((np.column_stack((X_new, data.columns)), np.column_stack((pred_calif,np.column_stack((pred_calif_pure, pred_calif_translate)))))), columns = list(data.loc[feat_key.index].index)+['Periodo', 'Rating Predicc', 'Rating Pure', 'Rating Local Trad'])
    print('Predicción Rating:')
    print('')
    print(data_pred[['Periodo', 'Rating Predicc', 'Rating Pure', 'Rating Local Trad']])

    # Output file:
    data_pred.to_csv(output_file, index = False)
    return(None)
