import numpy as num
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy import stats
from pyrocko.model import load_events
from matplotlib import pyplot as plt

'''
Note: Methods centroid, median and ward are correctly
      defined only if Euclidean pairwise metric is used.
'''

normalize = False
fn = sys.argv[1]
fn_events = '/home/marius/josef_dd/events_from_sebastian.pf'
method = 'weighted'       # works well for cc
# method = 'centroid'       # works well for cc
optimal_ordering = False

events = load_events(fn_events)
names_to_events = {e.name: e for e in events}

labels = []
data = []
with open(fn, 'r') as f:
    for line in f.readlines():
        e1, e2, cc = line.split()
        data.append(float(cc))
        labels.append(e2)

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

cut = fcluster(Z, 0.51, 'distance')

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.6])
print(len(Z))
Z1 = dendrogram(Z, orientation='right', labels=labels)
# for k, v in Z1.items():
#     print(k, v)
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
plt.show()
