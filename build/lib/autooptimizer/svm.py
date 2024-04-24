from sklearn.svm import SVC, SVR
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.simplefilter("ignore")

def svc(x, y, scoring='roc_auc_ovo'):
    def progress(cur, max):
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")

    xtrain ,xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    ss = StandardScaler()
    xtrain = ss.fit_transform(xtrain)
    xtest = ss.fit_transform(xtest)
    parameters={'kernel':['rbf','sigmoid','poly','linear'],
                'C':[np.arange(1,20,1),50,75,100,150,200,250,300,500,600,700,750,800,900,1e5],
                'degree':np.arange(0,20,0.5),
                'gamma':['scale','auto',np.arange(0.1,1,0.1), np.arange(1,10,1), np.arange(10,100,10)],
               'random_state':[4,12,20,22,40,42]}
    main_model = SVC(probability=True)
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(1, 21):
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
    return SVC(kernel=hyperparameter['kernel'], C=hyperparameter['C'], 
               degree=hyperparameter['degree'], gamma=hyperparameter['gamma'],
              random_state=hyperparameter['random_state'])

def svr(x, y , scoring='neg_mean_squared_error'):
    def progress(cur, max):
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")

    xtrain ,xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    ss = StandardScaler()
    xtrain = ss.fit_transform(xtrain)
    xtest = ss.fit_transform(xtest)
    parameters={'kernel':['rbf','sigmoid','poly','linear'],
                'C':[np.arange(1,20,1),50,75,100,150,200,250,300,500,600,700,750,800,900],
                'degree':np.arange(0,20,1),
                'gamma':['scale','auto',np.arange(0.1,1,0.1), np.arange(1,10,2), np.arange(10,100,20)],
                'epsilon': np.arange(0.1,0.5,0.1)}
    main_model = SVR()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(1, 21):
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
    return SVR(kernel=hyperparameter['kernel'], C=hyperparameter['C'], 
                degree=hyperparameter['degree'], gamma=hyperparameter['gamma'],
                epsilon=hyperparameter['epsilon'])
