import numpy as np
import matplotlib.pyplot as plt 
import time   
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
 
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
    
    clustering_names = 'MeanShift'
    
    plt.figure(figsize=(2 + 3, 9.5))
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
        #ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms = cluster.MeanShift(bandwidth=bandwidth)
        
    
        # predict cluster memberships
        t0 = time.time()
        ms.fit(X)
        t1 = time.time()
        if hasattr(ms, 'labels_'):
            y_pred = ms.labels_.astype(np.int)
        else:
            y_pred = ms.predict(X)

        # plot
        plt.subplot(4, 1, plot_num)
        if i_dataset == 0:
            plt.title(clustering_names, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(ms, 'cluster_centers_'):
            centers = ms.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15, horizontalalignment='right')
        plot_num += 1 
    plt.show()