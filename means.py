import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time   
from sklearn.datasets import make_blobs
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    '''
    cluster1=np.random.uniform(0.5,1.5,(2,10))
    cluster2=np.random.uniform(3.5,4.5,(2,10))
    X=np.hstack((cluster1,cluster2)).T
    
    plt.figure()
    plt.axis([0,5,0,5])
    plt.grid(True)
    plt.plot(X[:,0],X[:,1],'k.')
    
    K=range(1,10)
    meandistortions=[]
    for k in K:
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])
    
    plt.plot(K,meandistortions,'bx-');
    plt.xlabel('K');
    plt.ylabel('distortion');    
    plt.show()
    '''
    
    '''
    plt.figure(figsize=(12, 12))
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=1, random_state=random_state).fit_predict(X)
    k = KMeans(n_clusters=1, random_state=random_state)
    k.fit(X)
    plt.subplot(221)  #在2图里添加子图1
    plt.scatter(X[:, 0], X[:, 1], c=y_pred) #scatter绘制散点
    plt.title("Incorrect Number of Blobs")   #加标题
    plt.plot(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], 'ro')
    
    
    # Anisotropicly distributed data
    transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)    #返回的是乘积的形式
    y_pred = KMeans(n_clusters=1, random_state=random_state).fit_predict(X_aniso)
    k = KMeans(n_clusters=1, random_state=random_state)
    k.fit(X_aniso)    
    plt.subplot(222)#在2图里添加子图2
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
    plt.title("Anisotropicly Distributed Blobs")
    plt.plot(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], 'ro')
    
    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    y_pred = KMeans(n_clusters=1, random_state=random_state).fit_predict(X_varied)
    k = KMeans(n_clusters=1, random_state=random_state)
    k.fit(X_varied)    
    plt.subplot(223)#在2图里添加子图3
    plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
    plt.title("Unequal Variance")
    plt.plot(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], 'ro')
    
    # Unevenly sized blobs
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=1, random_state=random_state).fit_predict(X_filtered)
    k = KMeans(n_clusters=1, random_state=random_state)
    k.fit(X_filtered)
    plt.subplot(224)#在2图里添加子图4
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
    plt.title("Unevenly Sized Blobs")
    plt.plot(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], 'ro')
    plt.show() #显示图    
    '''
    

    
    np.random.seed(0)
    
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    n_samples = 50
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    clustering_names = [
        'MiniBatchKMeans', 'MeanShift', 'DBSCAN', 'Birch']
    
    plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    plot_num = 1
  
    datasets = [noisy_circles, noisy_moons, blobs, no_structure]
    for i_dataset, dataset in enumerate(datasets):
        X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
    
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.5)
        # create clustering estimators
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        
        two_means = cluster.MiniBatchKMeans(n_clusters=2)
               
        dbscan = cluster.DBSCAN(eps=.2)

        birch = cluster.Birch(n_clusters=2)
        
        clustering_algorithms = [two_means,  ms, dbscan, birch]
    
        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
    
            # plot
            plt.subplot(4, len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    
            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15, horizontalalignment='right')
            plot_num += 1 
    plt.show()