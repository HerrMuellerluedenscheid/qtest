import numpy as num
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy import stats
from pyrocko.model import load_events, dump_events
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

    print('L1', labels)
    Z1 = dendrogram(Z, orientation='right', labels=labels)
    ax1.set_xticks([])

    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    Z2 = dendrogram(Z)
    print(Z1)
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

    print('L2', labels)
    clusters = {}
    for ename, color in zip(Z1['ivl'], Z1['color_list']):
        import pdb
        pdb.set_trace()
        c = clusters.get(color, [])
        c.append(marker.EventMarker(names_to_events[ename]))
        clusters[color] = c
    print('found %s clusters:' % len(clusters))
    for icluster, events in clusters.items():
        print(icluster, 'xxxxxxxx')
        for e in events:
            print(e.get_event().name)
    return clusters.values()


def get_clustered_events(Z, events):
    cut = fcluster(Z, 1.4, 'distance')
    clusters = []
    for i_cluster in num.unique(cut):
        mask = (i_cluster == cut)
        cluster_events = [events[i] for i in num.where(mask)[0]]
        clusters.append(cluster_events)
    print('found %s clusters' % len(clusters))
    return clusters


def get_clustered_events_by_dendrogram(Z, events, labels):
    name_to_events = {e.get_event().name: e for e in events}
    print('L2', labels)
    d = dendrogram(Z, no_plot=True, labels=labels)
    clusters = {}
    for ename, color in zip(d['ivl'], d['color_list']):
        c = clusters.get(color, [])
        c.append(marker.EventMarker(names_to_events[ename]))
        clusters[color] = c
    print('found %s clusters:' % len(clusters))
    for icluster, events in clusters.items():
        print(icluster, 'xxxxxxxx')
        for e in events:
            print(e.get_event().name)
    return clusters.values()


def set_attribute_by_cluster(clusters, attribute):
    f = 1 if attribute == 'kind' else 1000
    for i, events in enumerate(clusters):
        for e in events:
            setattr(e, attribute, (i+1)*f)


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

    data = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            e1, e2, cc = line.split()
            data.append(float(cc))

    y = num.array(data)

    if normalize:
        y /= num.max(y)
        print('normalize with factor %1.4f' % num.max(y))

    print('len(y): %s, min(y): %s, max(y): %s' % (len(y), y.min(), y.max()))
    print('mean(y): %1.4f' % num.mean(y))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(squareform(y), cmap=plt.cm.YlGnBu, origin='lower')
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)

    Z = linkage(y, method)
    make_cluster_matrix(Z, event_names)

    event_markers = [marker.EventMarker(e) for e in events]
    #  event_clusters = get_clustered_events(Z, event_markers)
    event_clusters = get_clustered_events_by_dendrogram(Z, event_markers, event_names)
    # make_cluster_map(event_clusters)
    set_attribute_by_cluster(event_clusters, 'kind')
    events_clustered = flatten(event_clusters)
    # dump_events(events_clustered, 'events_hirarchy_clustered.pf')
    gui.marker.save_markers(events_clustered, 'event_markers_clustered.pf')
    plt.show()
