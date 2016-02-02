import numpy as num
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from pyrocko import cake, gf, model, parimap
from pyrocko import orthodrome as ortho
import progressbar


class UniqueColor():
    def __init__(self, color_map=mpl.cm.jet, tracers=None):
        self.tracers = tracers
        self.color_map = color_map
        self.mapping = dict(zip(self.tracers, num.linspace(0, 1, len(self.tracers))))

    def __getitem__(self, tracer):
        return self.color_map(self.mapping[tracer])


def animate(framenumber, ax, results, colormap):
    ax.clear()
    for r in results:
        s, e1, e2, segments = r
        plot_segments(segments, ax=ax, color=colormap[s])
    #ax.invert_zaxis()
    ax.set_xlim3d([-2000, 2000])
    ax.set_ylim3d([-2000, 2000])
    ax.set_zlim3d([6000, 14000])
    ax.set_xlabel('N [m]')
    ax.set_ylabel('E [m]')
    ax.set_zlabel('D [m]')
    ax.view_init(30, 1.8 * framenumber)
    return ax,


def make_animation(results, colormap):
    fig, ax = get_3d_ax()
    anim = animation.FuncAnimation(fig, animate, fargs=((ax, results, colormap)), frames=10, interval=15, blit=True)
    anim.save('coupled_rays.mp4', fps=1, extra_args=['-vcodec', 'libx264'])
    plt.show()


def get_3d_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.invert_zaxis()
    return fig, ax


def filter_pairs(results, threshold_pass_factor, min_travel_distance, ax=None):
    filtered = []
    for r in results:
        s, e1, e2, segments = r
        if segments == None:
            continue

        last_segment, traveled_distance, passing_distance = segments[-1]
        gamma = traveled_distance/passing_distance
        if gamma<threshold_pass_factor or num.isnan(traveled_distance) or traveled_distance<min_travel_distance:
            continue
        else:
            filtered.append(r)
    print '%s of %s pairs passed' %(len(filtered), len(results))

    return filtered


def array_center(stations):
    lats, lons = [], []
    for s in stations:
        lats.append(s.lat)
        lons.append(s.lon)
    return gf.meta.Location(lat=num.mean(lats), lon=num.mean(lons))


def project2enz(arrival, azimuth_deg):
    azimuth = azimuth_deg*cake.d2r
    z, y, t = arrival.zxt_path_subdivided()
    e = num.sin(azimuth) * y[0]
    n = num.cos(azimuth) * y[0]
    return e, n, z[0]


def plot_segments(segments, ax=None, color='r'):
    if not ax:
        fig, ax = get_3d_ax()
    for i, segment in enumerate(segments):
        (x,y,z), traveled_dist, passing_dist = segment
        ax.plot(x, y, z, c=color, alpha=0.1, linewidth=2)


def stats_by_station(results):
    hit_counter = {}
    for r in results:
        s, e1, e2, segments = r
        if not s in hit_counter:
            hit_counter[s] = 1
        else:
            hit_counter[s] += 1

    return hit_counter


def hitcount_pie(hitcounter, colormap):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts = []
    labels = []
    colors = []
    for s, c in hitcounter.items():
        counts.append(c)
        labels.append('.'.join(s.nsl()))
        colors.append(colormap[s])
    ax.pie(counts, labels=labels)
    plt.show()


def xyz_from_hitcount(hit_counter):
    X = num.zeros(len(hit_counter))
    Y = num.zeros(len(hit_counter))
    Z = num.zeros(len(hit_counter))

    for i, k in enumerate(hit_counter.keys()):
        X[i] = k.lon
        Y[i] = k.lat
        Z[i] = hit_counter[k]

    return X, Y, Z


def xyz_from_stations(stations):
    X = []
    Y = []
    labels = []
    for s in stations:
        yield s.lon, s.lat, '.'.join(s.nsl())


def hitcount_map_from_file(filename, stations=None, save_to='hitcount_map.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y, Z = num.loadtxt(filename).T
    ax.scatter(X, Y, s=Z/num.max(Z)*60)
    if stations:
        for x, y, label in xyz_from_stations(stations):
            ax.plot(x, y, 'g^')
            ax.text(x, y, label,
                    bbox={'facecolor':'white', 'alpha':0.5, 'pad':0.1, 'edgecolor':'white'})
    ax.set_xlabel('longitude [$^\circ$]')
    ax.set_ylabel(save_to)
    fig.savefig('hitcount_map.png')
    plt.show()


def hitcount_map(hit_counter, stations, save_to='hitcount_map.png'):
    #data = num.zeros(len(hit_counter)*3).reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y, Z = xyz_from_hitcount(hit_counter)
    ax.scatter(X, Y, s=Z/num.max(Z)*60)

    for x, y, label in xyz_from_stations(stations):
        ax.plot(x, y, 'g^')
        ax.text(x, y, label,
                bbox={'facecolor':'white', 'alpha':0.5, 'pad':0.1, 'edgecolor':'white'})
    ax.set_xlabel('longitude [$^\circ$]')
    ax.set_ylabel(save_to)
    fig.savefig('hitcount_map.png')
    plt.show()


def get_passing_distance(ray_points, x0):
    segments = []
    traveled_distance = 0
    passing_distance = False
    for i in xrange(len(ray_points)-1):
        x1, x2 = ray_points[i:i+2]
        u = x2-x1
        u_norm = num.linalg.norm(u)
        v = x0-x1
        w = x0-x2
        p = num.dot(v, u) / u_norm**2
        if p >= 0 and p <= 1:
            traveled_distance += u_norm
            seg = num.column_stack([x1, x1+u])
        else:
            passing_distance = num.linalg.norm(num.cross(v, w))/u_norm
            projected_on_u = p * u
            seg = num.column_stack([x1, x1+projected_on_u])
            traveled_distance += num.linalg.norm(projected_on_u)
        segments.append((seg, traveled_distance, passing_distance))
        if passing_distance != False:
            break

    return passing_distance, segments


def make_station_grid(stations, num_n=20, num_e=20, edge_stretch=1.):
    lats, lons = [], []
    for s in stations:
        lats.append(s.lat)
        lons.append(s.lon)

    maxn = num.max(lats)
    minn = num.min(lats)
    maxe = num.max(lons)
    mine = num.min(lons)

    nrange = maxn-minn
    erange = maxe-mine

    nlons = num.repeat(num.linspace(mine-erange*edge_stretch,
                                    maxe+erange*edge_stretch, num_e), num_n)
    nlats = num.tile(num.linspace(minn-nrange*edge_stretch,
                                  maxn+nrange*edge_stretch, num_n), num_e)
    stations = []
    for i, (lat, lon) in enumerate(zip(nlats, nlons)):
        stations.append(model.Station(lat=lat, lon=lon, station='%i'%i))

    return stations


def plot_stations(stations):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for s in stations:
        ax.plot(s.lon, s.lat, 'bo')
    plt.show()


def process_stations(args):
    e, s, phases, east_m, north_m, compare_events, center = args
    _results = []
    d = ortho.distance_accurate50m(e, s)*cake.m2d
    arrival = earthmodel.arrivals([d], phases=phases, zstart=e.depth)
    if len(arrival)!=1:
        return [(e, s, 'len(arrival)!=1')]

    x, y, z = project2enz(arrival[0], ortho.azimuth(e, s))
    x = x * cake.d2m + east_m
    y = y * cake.d2m + north_m
    points_of_segments = num.column_stack([x, y, z])
    for cmp_e in compare_events:
        cmp_north_m, cmp_east_m = ortho.latlon_to_ne(center, cmp_e)
        passed, segments = get_passing_distance(points_of_segments, (cmp_east_m, cmp_north_m, cmp_e.depth))
        if passed:
            _results.append((s, e, cmp_e, segments))
        else:
            pass
    return _results


def process_events(args):
    center, s, e, points_of_segments, cmp_e = args
    cmp_north_m, cmp_east_m = ortho.latlon_to_ne(center, cmp_e)
    passed, segments = get_passing_distance(points_of_segments, (cmp_east_m, cmp_north_m, cmp_e.depth))
    if passed:
        return (s, e, cmp_e, segments)
    else:
        return ()

def process(sources, targets, earthmodel, phases):
    results = []
    center = array_center(sources)
    pb = progressbar.ProgressBar(maxval=len(sources)).start()
    i = 0
    ne_from_center = []
    for s in sources:
        ne_from_center.append(ortho.latlon_to_ne_numpy(center.lat, center.lon, *s.effective_latlon))

    for i_e, e in enumerate(sources):
        # can be cached:
        north_m, east_m = ne_from_center[i_e]
        #args = [(e, s, phases, east_m, north_m, compare_events, center) for s in stations]
        #tmp_results = pool.map(process_stations, args)
        #results.extend(tmp_results)

        for i_t, t in enumerate(targets):
            #d = ortho.distance_accurate50m(e, s)*cake.m2d
            arrival = earthmodel.arrivals([t.distance_to(e)*cake.m2d], phases=phases, zstart=e.depth)
            try:
                x, y, z = project2enz(arrival[0], e.azibazi_to(t)[0]) #ortho.azimuth(e, t))
            except IndexError as err:
                print err
                continue
            x = x * cake.d2m + east_m
            y = y * cake.d2m + north_m
            points_of_segments = num.column_stack([x, y, z])
            #tmp_results = parimap.parimap(process_events, [(center, s, e, points_of_segments, cmp_e) for cmp_e in compare_events])
            #results.extend(tmp_results)
            for i_cmp_e, cmp_e in enumerate(sources):
                cmp_north_m, cmp_east_m = ne_from_center[i_cmp_e]
                passed, segments = get_passing_distance(points_of_segments, num.array((cmp_east_m, cmp_north_m, cmp_e.depth)))
                results.append((t, e, cmp_e, segments))
        pb.update(i)
        i += 1
    pb.finish()
    return results

if __name__=='__main__':
    events = list(model.Event.load_catalog('/data/meta/events2008.pf'))
    compare_events = list(model.Event.load_catalog('/data/meta/events2008.pf'))
    webnet_stations = model.load_stations('/data/meta/stations.pf')
    #hitcount_map_from_file(filename='hitcount.txt', stations=webnet_stations)
    stations = make_station_grid(webnet_stations, num_n=20, num_e=20, edge_stretch=0.15)
    plot_stations(stations)
    colormap = UniqueColor(tracers=stations)
    phases = [cake.PhaseDef('p'), cake.PhaseDef('P')]
    earthmodel = cake.load_model('earthmodel_malek_alexandrakis.nd')

    # sollte besser in 2d gemacht werden. Dauert ja sonst viel laenger...
    fig, ax = get_3d_ax()

    results = process(events, stations, earthmodel, phases)

    filtered = filter_pairs(results, 5, 200., ax=ax)
    hitcount = stats_by_station(filtered)
    X, Y, Z = xyz_from_hitcount(hitcount)
    num.savetxt('hitcount.txt', num.column_stack((X, Y, Z)))
    hitcount_map(hitcount, webnet_stations)
    make_animation(filtered, colormap)

