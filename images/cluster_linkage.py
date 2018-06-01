import numpy as num
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy import stats
from pyrocko.model import load_events, dump_events
from matplotlib import pyplot as plt
from pyrocko import automap, gmtpy

'''
Note: Methods centroid, median and ward are correctly
      defined only if Euclidean pairwise metric is used.
'''

def flatten(clusters):
    flattened = []
    for c in clusters:
        flattened.extend(c)
    return flattened


def make_cluster_map(clusters):
    
    clat = []
    clon = []
    for e in flatten(clusters):
        clat.append(e.lat)
        clon.append(e.lon)
    clat = num.mean(clat)
    clon = num.mean(clon)

    amap = automap.Map(radius=10, lat=clat, lon=clon)

    for events in clusters:
        lons = [e.lon for e in events]
        lats = [e.lat for e in events]

        # print(amap.gmt.jxyr())
        # amap.gmt.psxy(
        #       in_columns=(lons, lats),
        #       S='t10p',
        #       G='black',
        #       *amap.gmt.jxyr())



def make_cluster_matrix(Z, labels):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.6])

    Z1 = dendrogram(Z, orientation='right', labels=event_names)
    ax1.set_xticks([])

    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    Z2 = dendrogram(Z)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # heatmap:
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']

    D = squareform(y)
    D = D[idx1, :]
    D = D[:, idx2]

    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_yticks([])
    axmatrix.set_xticks([])

    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)


def get_clustered_events(Z, events):
    cut = fcluster(Z, 0.8, 'distance')
    clusters = []
    for i_cluster in num.unique(cut):
        mask = (i_cluster == cut)
        cluster_events = [events[i] for i in num.where(mask)[0]]
        clusters.append(cluster_events)

    return clusters


def set_depth_by_cluster(clusters):
    for i, events in enumerate(clusters):
        for e in events:
            e.depth = (i+1)*1000


if __name__ == '__main__':
    normalize = False
    fn = sys.argv[1]
    fn_events = sys.argv[2]
    
    
    # method = 'weighted'       # works well for cc
    method = 'complete'       # works well for cc
    optimal_ordering = False

    events = load_events(fn_events)
    event_names = [e.name for e in events]
    names_to_events = {e.name: e for e in events}

    labels = []
    data = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            e1, e2, cc = line.split()
            data.append(float(cc))
            labels.append((e1, e2))

    y = num.array(data)

    if normalize:
        y /= num.max(y)
        print('normalize with factor %1.4f' % num.max(y))

    print('len(y): %s, min(y): %s, max(y): %s' % (len(y), y.min(), y.max()))
    print('n labels: %s' % (len(labels)))
    print('mean(y): %1.4f' % num.mean(y))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(squareform(y), cmap=plt.cm.YlGnBu, origin='lower')
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)

    Z = linkage(y, method)
    make_cluster_matrix(Z, labels)

    event_clusters = get_clustered_events(Z, events)

    # make_cluster_map(event_clusters)
    set_depth_by_cluster(event_clusters)
    events_clustered = flatten(event_clusters)
    dump_events(events_clustered, 'events_hirarchy_clustered.pf')
    plt.show()
