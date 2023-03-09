from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import numpy as np

def kmeans(x):
    def progress(cur, max):
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")

    x = np.array(x)
    silhouette_score = np.array([])
    n_clusters_ = np.array([])
    algorithm = ['auto']
    for n in range(2, 21):
        model = KMeans(n_clusters=n).fit(x)
        label = model.labels_
        unique_labels = np.unique(label)
        if len(unique_labels) > 1:
            silhouette_score = np.append(silhouette_score, metrics.silhouette_score(x, label))
            n_clusters_ = np.append(n_clusters_, n)
        progress(n, 20)
    best_index = np.argmax(silhouette_score)
    best_score = silhouette_score[best_index]
    best_cluster = n_clusters_[best_index]
    silhouette_score = np.array([])
    for a in algorithm:
        model = KMeans(n_clusters=int(best_cluster), algorithm=a).fit(x)
        label = model.labels_
        silhouette_score = np.append(silhouette_score, metrics.silhouette_score(x, label))
    best_algo = algorithm[np.argmax(silhouette_score)]
    optimized_model = KMeans(n_clusters=int(best_cluster), algorithm=best_algo)
    print('Optimized is compeleted')
    print('The best score based on silhouette metric is {}'.format(round(best_score, 2)))
    return optimized_model

def meanshift(x):
    def progress(cur, max):
        cur *= 10
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")

    x = np.array(x)
    scores = np.array([])
    quantile_ = np.array([])
    for q in np.arange(0.1, 1.1, 0.1):
        bandwidth = estimate_bandwidth(x, quantile=q, random_state=4)
        model = MeanShift(bandwidth=bandwidth).fit(x)
        label = model.labels_
        unique_labels = np.unique(label)
        if len(unique_labels) > 1:
            silhouette = metrics.silhouette_score(x,label)
            scores = np.append(scores, silhouette)
            quantile_ = np.append(quantile_, q)
        progress(q, 10)
    best_index = np.argmax(scores)
    best_score = scores[best_index]
    best_quantile = quantile_[best_index]
    optimized_bandwith = estimate_bandwidth(x, quantile=best_quantile, random_state=4)
    optimized_model = MeanShift(bandwidth=optimized_bandwith)
    print('Optimized is compeleted')
    print('The best score based on silhouette metric is {}'.format(round(best_score, 2)))
    return optimized_model

def dbscan(x):
    def progress(cur, max):
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")

    x = np.array(x)
    silhouette = np.array([])
    distances = np.array([])
    min_sample_ = np.array([])
    for n in range(2, 21):
        NN = NearestNeighbors(n_neighbors=n).fit(x)
        distance, indice = NN.kneighbors(x)
        eps = distance.max()
        model = DBSCAN(eps=eps).fit(x)
        label = model.labels_
        unique_labels = np.unique(label)
        if len(unique_labels) > 1:
            silhouette = np.append(silhouette, metrics.silhouette_score(x, label))
            distances = np.append(distances, eps)
        progress(n,20)
    best_index = np.argmax(silhouette)
    best_eps = distances[best_index]
    silhouette = np.array([])
    for m in range(2,20):
        model = DBSCAN(eps=best_eps, min_samples=m).fit(x)
        label = model.labels_
        silhouette = np.append(silhouette, metrics.silhouette_score(x, label))
        min_sample_ = np.append(min_sample_, m)
    best_sample = min_sample_[np.argmax(silhouette)]
    best_score = silhouette[np.argmax(silhouette)]
    optimized_model = DBSCAN(eps=best_eps, min_samples=int(best_sample))
    print('Optimized is compeleted')
    print('The best score based on silhouette metric is {}'.format(round(best_score, 2)))
    return optimized_model

def minibatchkmeans(x):
    def progress(cur, max):
        p = round(100*cur/max)
        banner = "Optimizing in Progress: ["+'|'*int(p/5)+" "*(20-int(p/5))+"] -- {0}%".format(float(p))
        print(banner, end="\r")
    
    x = np.array(x)
    silhouette_score = np.array([])
    n_clusters_ = np.array([])
    random_state_ = np.array([0, 1, 3, 4, 11, 14, 20, 40, 43])
    batch_size_ = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    for n in range(2, 21):
        model = MiniBatchKMeans(n_clusters=n).fit(x)
        label = model.labels_
        unique_labels = np.unique(label)
        if len(unique_labels) > 1:
            silhouette_score = np.append(silhouette_score, metrics.silhouette_score(x, label))
            n_clusters_ = np.append(n_clusters_, n)
        progress(n,20)
    best_index = np.argmax(silhouette_score)
    best_score = silhouette_score[best_index]
    best_cluster = n_clusters_[best_index]
    silhouette_score = np.array([])
    for r in random_state_:
        model = MiniBatchKMeans(n_clusters=int(best_cluster), random_state=r).fit(x)
        label = model.labels_ 
        silhouette_score = np.append(silhouette_score, metrics.silhouette_score(x, label))
    best_random_state_ = random_state_[np.argmax(silhouette_score)]
    silhouette_score = np.array([])
    for b in batch_size_:
        model = MiniBatchKMeans(n_clusters=int(best_cluster), 
                                random_state=best_random_state_, batch_size=b).fit(x)
        label = model.labels_
        silhouette_score = np.append(silhouette_score, metrics.silhouette_score(x, label))
    best_batch_size = batch_size_[np.argmax(silhouette_score)]
    optimized_model = MiniBatchKMeans(n_clusters=int(best_cluster), random_state=best_random_state_,
                                        batch_size=best_batch_size)
    print('Optimized is compeleted')
    print('The best score based on silhouette metric is {}'.format(round(best_score, 2)))
    return optimized_model