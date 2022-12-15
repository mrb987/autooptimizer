from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import warnings
warnings.simplefilter('ignore')

def decisiontreeclassifier(x, y, scoring='roc_auc_ovo'):
    xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'max_depth':range(1,20,1), 'random_state':range(1,50,1), 'max_features':['auto', 'sqrt', 'log2'],
                'splitter':['best', 'random'], 'criterion':['gini', 'entropy', 'log_loss'],
                 'min_samples_split':np.arange(1,20,1),'min_samples_leaf': np.arange(1,20,1),
                 'min_weight_fraction_leaf':[0,0.5]}
    main_model = DecisionTreeClassifier()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4, 11):
        scv_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        scv_model.fit(xtrain,ytrain)
        if scv_model.best_score_ != 1.:
            best_params=np.append(best_params,scv_model.best_params_)
            best_score=np.append(best_score,scv_model.best_score_)
    best_index = np.argmax(best_score)
    print('The best possible accuracy in terms of {0} metric is {1}%'.format(scoring, round(best_score[best_index] * 100, 2)))
    hyperparameter =  best_params[best_index]
    return DecisionTreeClassifier(max_depth=hyperparameter['max_depth'] , random_state=hyperparameter['random_state'], 
                                max_features=hyperparameter['max_features'], splitter=hyperparameter['splitter'],
                                criterion=hyperparameter['criterion'],min_samples_split=hyperparameter['min_samples_split'],
                                min_samples_leaf=hyperparameter['min_samples_leaf'], min_weight_fraction_leaf=hyperparameter['min_weight_fraction_leaf'])

def decisiontreeregressor(x, y, scoring='neg_mean_absolute_error'):
    xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.3, random_state=4, shuffle=True)
    parameters = {'max_depth':range(1,20,1), 'random_state':range(1,50,1), 'max_features':['auto', 'sqrt', 'log2'],
                'splitter':['best', 'random'], 'criterion':['friedman_mse', 'poisson'],
                 'min_samples_split':np.arange(1,20,1), 'min_samples_leaf': np.arange(1,20,1),
                 'min_weight_fraction_leaf':[0,0.5]}
    main_model = DecisionTreeRegressor()
    best_params = np.array([])
    best_score = np.array([])
    for cv in range(4, 11):
        scv_model = RandomizedSearchCV(main_model, parameters, scoring=scoring, cv=cv)
        scv_model.fit(xtrain,ytrain)
        if scv_model.best_score_ != 1.:
            best_params=np.append(best_params, scv_model.best_params_)
            best_score=np.append(best_score, scv_model.best_score_)
    best_index = np.argmax(best_score)
    print('The best possible accuracy in terms of {0} metric is {1}'.format(scoring, round(best_score[best_index], 4)))
    hyperparameter =  best_params[best_index]
    return DecisionTreeRegressor(max_depth=hyperparameter['max_depth'] , random_state=hyperparameter['random_state'], 
                                max_features=hyperparameter['max_features'], splitter=hyperparameter['splitter'],
                                criterion=hyperparameter['criterion'], min_samples_split=hyperparameter['min_samples_split'],
                                min_samples_leaf=hyperparameter['min_samples_leaf'], min_weight_fraction_leaf=hyperparameter['min_weight_fraction_leaf'])

