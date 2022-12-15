from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import warnings
warnings.simplefilter('ignore')

def randomforestclassifier(x, y, scoring='roc_auc_ovo'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'n_estimators': np.arange(10,50,5),'max_depth':np.arange(1,10,1),
                 'criterion':['entropy','gini'], 'random_state':[1,3,4,11,14,20,43]}
    main_model = RandomForestClassifier()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4, 11):
        svc_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        svc_model.fit(xtrain, ytrain)
        if svc_model.best_score_!=1.:
            best_params = np.append(best_params, svc_model.best_params_)
            best_score = np.append(best_score, svc_model.best_score_)
    best_index = np.argmax(best_score)
    print('The best possible accuracy in terms of {0} metric is {1}%'.format(scoring, round(best_score[best_index]*100,2)))
    hyperparameters = best_params[best_index]
    return RandomForestClassifier(n_estimators=hyperparameters['n_estimators'], max_depth=hyperparameters['max_depth'],
                                 criterion=hyperparameters['criterion'], random_state=hyperparameters['random_state'])

def gradientboostingclassifier(x, y, scoring='roc_auc_ovo'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'loss':['deviance', 'exponential'], 'learning_rate':np.arange(0.1,2, 0.2), 
                  'n_estimators':np.arange(10,100,10), 'random_state':[4,10,20,40,42],
                  'max_depth':np.arange(1,10,1), 'max_features':['auto','sqrt','log2'], 
                  'criterion':['friedman_mse','squared_error', 'mse','mae'], 'warm_start':[True, False]}
    main_model = GradientBoostingClassifier()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4, 11):
        svc_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        svc_model.fit(xtrain, ytrain)
        if svc_model.best_score_ != 1.:
            best_params = np.append(best_params, svc_model.best_params_) 
            best_score = np.append(best_score, svc_model.best_score_)
    best_index = np.argmax(best_score)
    hyperparameters = best_params[best_index]
    print('The best possible accuracy in terms of {0} metric is {1}%'.format(scoring, round(best_score[best_index]*100,2)))
    return GradientBoostingClassifier(loss=hyperparameters['loss'], learning_rate=hyperparameters['learning_rate'],
                                n_estimators=hyperparameters['n_estimators'], max_depth=hyperparameters['max_depth'],
                                max_features=hyperparameters['max_features'], criterion=hyperparameters['criterion'])

def gradientboostingregressor(x, y, scoring='neg_mean_absolute_error'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                  'learning_rate':np.arange(0.1,2, 0.2),'subsample':np.arange(0.0,1.0,0.1),
                  'n_estimators':np.arange(10,100,10), 'random_state':[4,10,20,40,42],
                  'max_depth':np.arange(1,20,1), 'max_features':['auto','sqrt','log2'], 
                  'criterion':['friedman_mse','squared_error', 'mse'], 'warm_start':[True, False],
                 'alpha':np.arange(0.0,1.0,0.1),'validation_fraction':np.arange(0.0,1.0,0.1)}
    main_model = GradientBoostingRegressor()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4, 11):
        svc_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        svc_model.fit(xtrain, ytrain)
        if svc_model.best_score_ != 1.:
            best_params = np.append(best_params, svc_model.best_params_) 
            best_score = np.append(best_score, svc_model.best_score_)
    best_index = np.argmax(best_score)
    hyperparameters = best_params[best_index]
    print('The best possible accuracy in terms of {0} metric is {1}'.format(scoring, round(best_score[best_index], 4)))
    return GradientBoostingRegressor(loss=hyperparameters['loss'], learning_rate=hyperparameters['learning_rate'],
                                subsample=hyperparameters['subsample'], n_estimators=hyperparameters['n_estimators'],
                                random_state=hyperparameters['random_state'], max_depth=hyperparameters['max_depth'],
                                     max_features=hyperparameters['max_features'], criterion=hyperparameters['criterion'],
                                     warm_start=hyperparameters['warm_start'], alpha=hyperparameters['alpha'],
                                     validation_fraction=hyperparameters['validation_fraction'])

def adaboostclassifier(x, y, scoring='roc_auc_ovo'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'base_estimator':[DecisionTreeClassifier(), KNeighborsClassifier(),
                                   SVC(), LogisticRegression(), GaussianNB()], 'n_estimators': np.arange(10,50,5),
                  'learning_rate': np.arange(0.1,2, 0.2),'algorithm': ['SAMME'], 'random_state':[1,3,4,11,14,20,43],}
    main_model = AdaBoostClassifier()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4,11):
        scv_model = RandomizedSearchCV(main_model, parameters, scoring=scoring)
        scv_model.fit(xtrain, ytrain)
        if scv_model.best_score_ != 1.:
            best_params = np.append(best_params, scv_model.best_params_)
            best_score = np.append(best_score, scv_model.best_score_)
    best_index = np.argmax(best_score)
    print('The best possible accuracy in terms of {0} metric is {1}%'.format(scoring, round(best_score[best_index]*100,2)))
    hyperparameters = best_params[best_index]
    return AdaBoostClassifier(base_estimator=hyperparameters['base_estimator'], n_estimators=hyperparameters['n_estimators'],
                              learning_rate=hyperparameters['learning_rate'],algorithm=hyperparameters['algorithm'], 
                              random_state=hyperparameters['random_state'])

def adaboostregressor(x, y, scoring='neg_mean_absolute_error'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'base_estimator':[KNeighborsRegressor(), SVR(), LinearRegression(),
                                   DecisionTreeRegressor()],
                 'n_estimators':np.arange(10,50,5), 'learning_rate':np.arange(0.1,2, 0.2),
                 'loss':['linear', 'square', 'exponential'], 'random_state':[4,10,20,40,42]}
    main_model = AdaBoostRegressor()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4,11):
        scv_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        scv_model.fit(xtrain, ytrain)
        if scv_model.best_score_ !=1.:
            best_score = np.append(best_score, scv_model.best_score_)
            best_params = np.append(best_params, scv_model.best_params_)
    best_index = np.argmax(best_score)
    print('The best possible accuracy in terms of {0} metric is {1}'.format(scoring, round(best_score[best_index], 4)))
    hyperparameters = best_params[best_index]
    return AdaBoostRegressor(base_estimator=hyperparameters['base_estimator'], n_estimators=hyperparameters['n_estimators'],
                              learning_rate=hyperparameters['learning_rate'], random_state=hyperparameters['random_state'],
                            loss=hyperparameters['loss'])

def baggingclassifier(x, y, scoring='roc_auc_ovo'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'base_estimator':[DecisionTreeClassifier(), KNeighborsClassifier(),
                                   SVC(), LogisticRegression(), GaussianNB()], 'n_estimators':np.arange(5,50,5),
                  'bootstrap':[True, False],'bootstrap_features':[True, False],
                  'oob_score':[True, False],'warm_start':[True, False],
                  'random_state':[4,11,20,40,42]}
    main_model = BaggingClassifier()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4,11):
        scv_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        scv_model.fit(xtrain, ytrain)
        if scv_model.best_score_ != 1.:
            best_params = np.append(best_params, scv_model.best_params_)
            best_score = np.append(best_score, scv_model.best_score_)
    best_index = np.argmax(best_score)
    print('The best possible accuracy in terms of {0} metric is {1}%'.format(scoring, round(best_score[best_index]*100,2)))
    hyperparameters = best_params[best_index]
    return BaggingClassifier(base_estimator=hyperparameters['base_estimator'], n_estimators=hyperparameters['n_estimators'],
                            bootstrap=hyperparameters['bootstrap'], bootstrap_features=hyperparameters['bootstrap_features'],
                            oob_score=hyperparameters['oob_score'], warm_start=hyperparameters['warm_start'],
                            random_state=hyperparameters['random_state'])

def baggingregressor(x, y, scoring='neg_mean_absolute_error'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'base_estimator':[DecisionTreeRegressor(), KNeighborsRegressor(),
                                   SVR(), LinearRegression()], 'n_estimators':np.arange(10,50,5),
                'bootstrap':[True, False], 'bootstrap_features':[True, False],
                 'oob_score':[True, False], 'warm_start':[True, False],
                 'random_state':[4,11,20,40,42]}
    main_model = BaggingRegressor()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4,11):
        scv_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        scv_model.fit(xtrain, ytrain)
        if scv_model.best_score_ !=1.:
            best_params = np.append(best_params, scv_model.best_params_)
            best_score = np.append(best_score, scv_model.best_score_)
    best_index = np.argmax(best_score)
    print('The best possible accuracy in terms of {0} metric is {1}'.format(scoring, round(best_score[best_index], 4)))
    hyperparameters = best_params[best_index]
    return BaggingRegressor(base_estimator=hyperparameters['base_estimator'], n_estimators=hyperparameters['n_estimators'],
                            bootstrap=hyperparameters['bootstrap'], bootstrap_features=hyperparameters['bootstrap_features'],
                            oob_score=hyperparameters['oob_score'], warm_start=hyperparameters['warm_start'],
                            random_state=hyperparameters['random_state'])

def extratreesclassifier(x, y, scoring='roc_auc_ovo'):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'n_estimators':np.arange(10,100,10), 'criterion':['gini','entropy','log_loss'],
                 'max_depth':np.arange(1,20,1),'bootstrap':[True,False], 'oob_score':[True,False],
                 'random_state':[1,3,4,11,14,20,43],'class_weight':['balanced','balanced_subsample']}
    main_model = ExtraTreesClassifier()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4,11):
        scv_model = RandomizedSearchCV(main_model, parameters, cv=cv, scoring=scoring)
        scv_model.fit(xtrain, ytrain)
        if scv_model.best_score_ !=1.:
            best_params = np.append(best_params, scv_model.best_params_)
            best_score = np.append(best_score, scv_model.best_score_)
    best_index = np.argmax(best_score)
    hyperparameters = best_params[best_index]
    print('The best possible accuracy in terms of {0} metric is {1}%'.format(scoring, round(best_score[best_index]*100,2)))
    return ExtraTreesClassifier(n_estimators=hyperparameters['n_estimators'], max_depth=hyperparameters['max_depth'],
                                 criterion=hyperparameters['criterion'], random_state=hyperparameters['random_state'],
                               bootstrap=hyperparameters['bootstrap'],oob_score=hyperparameters['oob_score'],
                               class_weight=hyperparameters['class_weight'])
