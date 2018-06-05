import numpy as num
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy import stats
from pyrocko.model import load_events
from matplotlib import pyplot as plt
from pyrocko import automap, gmtpy, gui
from pyrocko.gui import marker

'''
Note: Methods centroid, median and ward are correctly
      defined only if Euclidean pairwise metric is used.
'''

def flatten(clusters):
    flattened = []
    for c in clusters:
        flattened.extend(c)
    return flattened


def make_cluster_matrix(Z, labels, distance):
    '''
    :param Z: output of `linkage`
    :param labels: list of labels
    :param distance: cutoff distance for clustering
    '''
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.6])

    Z1 = dendrogram(Z, orientation='right', labels=labels)
    ax1.axvline(distance, color='grey')

    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    Z2 = dendrogram(Z)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axhline(distance, color='grey')

    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    D = squareform(y)
    D = D[Z1['leaves'], :]
    D = D[:, Z2['leaves']]

    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_yticks([])
    axmatrix.set_xticks([])

    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)


def get_clustered_events(Z, events, distance):
    cut = fcluster(Z, distance, 'distance')
    clusters = []
    for i_cluster in num.unique(cut):
        mask = (i_cluster == cut)
        cluster_events = [events[i] for i in num.where(mask)[0]]
        clusters.append(cluster_events)
    return clusters


def set_attribute_by_cluster(clusters, attribute):
    f = 1 if attribute == 'kind' else 1000
    for i, events in enumerate(clusters):
        for e in events:
            setattr(e, attribute, (i+1)*f)


def plot_distance_matrix(y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(squareform(y), cmap=plt.cm.YlGnBu, origin='lower')
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)


if __name__ == '__main__':
    fn = sys.argv[1]
    fn_events = sys.argv[2]

    # method = 'average'
    method = 'single'
    # method = 'weighted'
    # method = 'complete'
    distance = 0.442   # dist_cut for cluster selection

    events = load_events(fn_events)
    event_names = [e.name for e in events]
    names_to_events = {e.name: e for e in events}

    data = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            e1, e2, cc = line.split()
            data.append(float(cc))
    y = num.array(data)

    print('len(y): %s, min(y): %s, max(y): %s' % (len(y), y.min(), y.max()))
    print('mean(y): %1.4f' % num.mean(y))

    plot_distance_matrix(y)
    Z = linkage(y, method)
    make_cluster_matrix(Z, event_names, distance)

    event_markers = [marker.EventMarker(e) for e in events]
    event_clusters = get_clustered_events(Z, event_markers, distance=distance)
    print('found %s clusters (distance=%s)' % (len(event_clusters), distance))
    set_attribute_by_cluster(event_clusters, 'kind')
    events_clustered = flatten(event_clusters)
    gui.marker.save_markers(events_clustered, 'event_markers_clustered.pf')
    plt.show()
