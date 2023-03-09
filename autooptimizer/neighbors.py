from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import numpy as np
import warnings
warnings.simplefilter('ignore')

def kneighborsclassifier(x, y, scoring='roc_auc_ovo'):
    def progress(cur, max):
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'n_neighbors':range(1, 20),
                'weights':['uniform', 'distance'],
                'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric':['minkowski','manhattan','euclidean']}
    main_model = KNeighborsClassifier()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(1,21):
        scv_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv+1)
        scv_model.fit(xtrain, ytrain)
        if scv_model.best_score_ != 1.:
            best_params = np.append(best_params, scv_model.best_params_)
            best_score = np.append(best_score, scv_model.best_score_)
        progress(cv,20)
    best_index = np.argmax(best_score)
    print('Optimizing is completed')
    print('The best possible accuracy in terms of {0} metric is {1}%'.format(scoring, round(best_score[best_index] * 100, 2)))
    hyperparameter =  best_params[best_index]
    return KNeighborsClassifier(weights=hyperparameter['weights'], n_neighbors=hyperparameter['n_neighbors'], 
                                metric=hyperparameter['metric'], algorithm=hyperparameter['algorithm'])

def kneighborsregressor(x, y, scoring='neg_mean_squared_error'):
    def progress(cur, max):
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")
        
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'n_neighbors':range(1,20), 'weights':['uniform', 'distance'],
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'metric':['minkowski', 'manhattan', 'euclidean']}
    main_model = KNeighborsRegressor()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(1, 21):
        svc_model = RandomizedSearchCV(main_model, parameters, scoring=scoring ,cv=cv+1)
        svc_model.fit(xtrain, ytrain)
        if svc_model.best_score_ != 1.:
            best_params = np.append(best_params, svc_model.best_params_)
            best_score = np.append(best_score, svc_model.best_score_)
        progress(cv,20)
    best_index = np.argmax(best_score)
    print('Optimizing is completed')
    print('The best possible accuracy in terms of {0} metric is {1}'.format(scoring, round(best_score[best_index], 3)))
    hyperparameter = best_params[best_index]
    return KNeighborsRegressor(weights=hyperparameter['weights'], n_neighbors=hyperparameter['n_neighbors'], 
                                metric=hyperparameter['metric'], algorithm=hyperparameter['algorithm'])
