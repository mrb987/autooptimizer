from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def dbscan(x):
    NN=NearestNeighbors(n_neighbors=5).fit(x)
    distance, indice=NN.kneighbors(x)
    eps=distance.max()
    min_samples=x.ndim*2
    silhouette=np.array([])
    model=DBSCAN(eps=eps,min_samples=min_samples).fit(x)
    label=model.labels_
    silhouette=np.append(silhouette,metrics.silhouette_score(x,label))

    if silhouette < 0:
        print('''
        The score of clustring is {}%
        Datapoints have been assigned to the wrong clusters.
        You should consider preprocessing and dataset cleaning.
        and the parameters are:
        '''.format(round(silhouette[0]*100,2)))
        print('epsilon: ',eps)
        print('number of samples: ',min_samples)
    elif silhouette >= 0 and silhouette < 0.1:
        print('''
        The score of clustring is {}%
        ""overlapping clusters""
        You should consider preprocessing.
        and the parameters are:
        '''.format(round(silhouette[0]*100,2)))
        print('epsilon: ',eps)
        print('Optimal number of samples: ',min_samples)
    elif silhouette > 0.1 and silhouette < 0.5:
        print('''
        The score of clustring is {}%.
        Datapoints has been approximately assigned to the right clusters.
        and the parameters are:
        '''.format(round(silhouette[0]*100,2)))
        print('Best epsilon: ',eps)
        print('Optimal number of samples: ',min_samples)
    elif silhouette >= 0.5:
        print('''
        The score of clustring is {}%.
        Good clustering.
        and the parameters are:
        '''.format(round(silhouette[0]*100,2)))
        print('Best epsilon: ',eps)
        print('Optimal number of samples: ',min_samples)

def kmeans(x):
    calinski_harabasz_score=np.array([])
    silhouette_score=np.array([])
    for i in range(2,20):
        model=KMeans(n_clusters=i).fit(x)
        label=model.labels_
        silhouette_score=np.append(silhouette_score,metrics.silhouette_score(x,label))
        calinski_harabasz_score=np.append(calinski_harabasz_score,metrics.calinski_harabasz_score(x,label))
    best_score=np.argmax(calinski_harabasz_score)
    best_proba=calinski_harabasz_score[best_score]
    plt.style.use("fivethirtyeight")
    plt.figure(dpi=80,figsize=(9,5))
    plt.title('Model evaluation result')
    plt.grid()
    plt.margins(0.05)
    plt.xlabel('Number of clusters',fontsize=12,color='red')
    plt.ylabel('Score of evaluation',fontsize=12,color='red')
    plt.xticks(range(2, 20))
    plt.yticks()
    plt.annotate('Optimal clustering',xytext=(best_score+2,best_proba),xy=(best_score+2,best_proba),fontsize=9,arrowprops={'color':'violet'})
    plt.plot(range(2,20),calinski_harabasz_score,ls='--',marker='x',lw=2,label='calinski score')
    plt.grid()
    plt.legend(loc='best')
    plt.show()

def meanshift(x):
    bandwidth=estimate_bandwidth(x,quantile=0.3,random_state=2)
    model=MeanShift(bandwidth=bandwidth).fit(x)
    label=model.labels_
    silhouette=metrics.silhouette_score(x, label)
    if silhouette < 0:
        print('''
        The score of clustring is {}%
        Datapoints have been assigned to the wrong clusters.
        You should consider preprocessing and dataset cleaning.
        and the parameters are:
        '''.format(round(silhouette*100,2)))    
        print('the assigned bandwidth is: ',bandwidth)
    elif silhouette >= 0 and silhouette < 0.1:
        print('''
        The score of clustring is {}%.
        ""Overlapping clusters""
        You should consider preprocessing.
        and the parameters are:
        '''.format(round(silhouette*100,2)))
        print('the assigned bandwidth is: ',bandwidth)
    elif silhouette > 0.1 and silhouette <= 0.5:
        print('''
        The score of clustring is {}%.
        Datapoints have been approximately assigned to the right clusters
        and the parameters are:
        '''.format(round(silhouette*100,2)))
        print('the assigned bandwidth is: ',bandwidth)
    elif silhouette > 0.5:
        print('''
        The score of clustring is {}%.
        Good clustering
        and the parameters are:
        '''.format(round(silhouette*100,2)))
        print('The assigned bandwidth is:')
        return bandwidth
