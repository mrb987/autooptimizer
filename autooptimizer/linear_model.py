from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')

def logisticregression(x,y):
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=4,shuffle=True)
    parameters={'penalty':['l1','l2','elasticnet', 'none'],
                'C':[np.arange(0.1,10,0.1),1,10,100,1000],
                'random_state':range(1,50),
                'solver':['saga'],
                'max_iter':[1,10,100]}
    l1_parameters={'penalty':['l1','l2'],
                'C':[np.arange(0.1,10,0.1),1,10,100,1000],
                'random_state':range(1,50),
                'solver':['liblinear'],
                'max_iter':[1,10,100]}
    l2_parameters={'penalty':['l2', 'none'],
                'C':[np.arange(0.1,1,0.1),1,10,100,1000],
                'random_state':range(1,50),
                'solver':['newton-cg', 'lbfgs', 'sag',],
                'max_iter':[1,10,100]}
    main_model=LogisticRegression()
    scv_model1=RandomizedSearchCV(main_model,parameters,cv=5)
    scv_model2=RandomizedSearchCV(main_model,l1_parameters,cv=5)
    scv_model3=RandomizedSearchCV(main_model,l2_parameters,cv=5)
    best_params= np.array([])
    best_score=np.array([])
    for i in range(x.ndim*4):
        scv_model1.fit(xtrain,ytrain)
        best_params=np.append(best_params,scv_model1.best_params_)
        best_score=np.append(best_score,scv_model1.best_score_)
    for i in range(x.ndim*4):
        scv_model2.fit(xtrain,ytrain)
        best_params=np.append(best_params,scv_model2.best_params_)
        best_score=np.append(best_score,scv_model2.best_score_)
    for i in range(x.ndim*4):
        scv_model3.fit(xtrain,ytrain)
        best_params=np.append(best_params,scv_model3.best_params_)
        best_score=np.append(best_score,scv_model3.best_score_)

    best_index=np.argmax(best_score)
    print('The best accuracy depends on you dataset is {}%'.format(round(best_score[best_index]*100,2)))
    print('The parameters are: ')
    return best_params[best_index]
    
