from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import numpy as np

def kneighborsclassifier(x,y):
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2,shuffle=True)
    parameters={'n_neighbors':range(1,20),
                'weights':['uniform', 'distance'],
                'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric':['minkowski','manhattan','euclidean']}
    main_model=KNeighborsClassifier()
    scv_model=RandomizedSearchCV(main_model,parameters,cv=10)
    best_params=np.array([])
    best_score=np.array([])
    for i in range(x.ndim*3):
        scv_model.fit(xtrain,ytrain)
        best_params=np.append(best_params,scv_model.best_params_)
        best_score=np.append(best_score,scv_model.best_score_)
    best_index=np.argmax(best_score)
    print('The best accuracy depends on you dataset is {}%'.format(round(best_score[best_index]*100,2)))
    print('The parameters are: ')
    return best_params[best_index]
