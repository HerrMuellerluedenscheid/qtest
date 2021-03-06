import numpy as num
from scipy.interpolate import griddata
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from pyrocko import cake, gf, model, parimap
from pyrocko import orthodrome as ortho
from pyrocko import util
from pyrocko.gf import ExplosionSource, Source, DCSource, Target, meta, SourceWithMagnitude
from pyrocko.guts import List, Object, String, Tuple, Float
import progressbar
import logging
import sys
logger = logging.getLogger()


cmap = plt.cm.bone_r


class UniqueColor():
    def __init__(self, color_map=mpl.cm.jet, tracers=None):
        self.tracers = tracers
        self.color_map = color_map
        self.mapping = dict(zip(self.tracers, num.linspace(0, 1, len(self.tracers))))

    def __getitem__(self, tracer):
        return self.color_map(self.mapping[tracer])


class Hookup():
    def __init__(self, reference):
        self.reference = reference

    @classmethod
    def from_locations(cls, locations):
        return cls(array_center(locations))

    def __call__(self, location):
        north_m, east_m = ortho.latlon_to_ne_numpy(self.reference.effective_lat,
                                             self.reference.effective_lon,
                                             location.effective_lat,
                                             location.effective_lon)
        #north_m, east_m = ortho.latlon_to_ne(self.reference, location)
        return num.array(((north_m[0], east_m[0], location.depth-self.reference.depth), )).T
        #return num.array(((north_m[0], east_m[0], 0),)).T

    def add_to_ned(self, from_location, ned):
        correction = self(from_location)
        correction[-1] -= from_location.depth
        ned = num.array(ned) + correction
        return ned


class Filtrate(Object):
    sources = List.T(SourceWithMagnitude.T())
    targets = List.T(Target.T())
    earthmodel = meta.Earthmodel1D.T()
    phases = List.T(String.T())
    #couples = List.T(List.T(DCSource.T(),
    #                         DCSource.T(),
    #                         Target.T(),
    #                         Float.T(),
    #                         Float.T()), help="(s1, s2, t, passing_distance, traveled_distance)")
    couples = List.T(List.T())
    def __iter__(self):
        return iter(self.couples)

    def __len__(self):
        return len(self.couples)

class Coupler():
    def __init__(self, filtrate=None):
        self.results = []
        self.hookup = None
        self.filtrate = filtrate
        print 'phases in process obsolete'

    def process(self, sources, targets, earthmodel, phases, ignore_segments=True, dump_to=False, check_relevance_by=False):
        self.filtrate = Filtrate(sources=sources, targets=targets, phases=phases, earthmodel=earthmodel)
        phases = [cake.PhaseDef('p'), cake.PhaseDef('P')]
        i = 0
        self.hookup = Hookup.from_locations(targets)
        failed = 0
        passed = 0
        logger.info('start coupling events')
        widgets = ['event couples ', progressbar.Percentage(), progressbar.Bar()]
        pb = progressbar.ProgressBar(maxval=len(sources), widgets=widgets).start()
        for i_e, ev in enumerate(sources):
            pb.update(i_e)
            for i_t, t in enumerate(targets):
                if not self.is_relevant(ev, t, check_relevance_by):
                    continue

                arrival = earthmodel.arrivals([t.distance_to(ev)*cake.m2d], phases=phases, zstart=ev.depth)
                try:
                    #incidence_angle = arrival[0].incidence_angle()
                    n, e, d = project2enz(arrival[0], ev.azibazi_to(t)[0])
                except IndexError as err:
                    print '!!!Error> ', err
                    continue

                n = n * cake.d2m
                e = e * cake.d2m
                points_of_segments = self.hookup.add_to_ned(ev, (n,e,d))
                for i_cmp_e, cmp_e in enumerate(sources):
                    ned_cmp = self.hookup(cmp_e)
                    try:
                        traveled_distance, passing_distance, segments = self.get_passing_distance(
                            points_of_segments, num.array(ned_cmp))
                        passed += 1
                    except IndexError:
                        failed += 1
                        continue
                    #if ignore_segments:
                    #    segments = []
                    if traveled_distance:
                        self.filtrate.couples.append([
                            ev, cmp_e, t, float(traveled_distance),
                            float(passing_distance),
                            #float(self.ray_length(arrival[0])),
                            None,
                            #float(incidence_angle)])
                            None])
                    #self.results.append(
                    #    (ev, cmp_e, t, traveled_distance, passing_distance,
                    #     segments, incidence_angle))
            pb.update(i)
            i += 1
        pb.finish()
        print 'failed: %s, passed:%s ' % (failed, passed)
        if passed == 0:
            raise Exception('Coupling failed')

        if dump_to:
            self.filtrate.validate()
            self.filtrate.dump(filename=dump_to)

    def ray_length(self, arrival):
        z, x, t = arrival.zxt_path_subdivided()
        return num.sum(num.sqrt(num.diff(num.array(x)*cake.d2m)**2+num.diff(num.array(z))**2))


    def is_relevant(self, source, target, p):
        if p==False or p ==None:
            return True
        else:
            return p.relevant(
                source.time, source.time+20,
                trace_selector=lambda x: (x.station==target.codes[1]))

    def filter_pairs(self, threshold_pass_factor, min_travel_distance, data, ignore=[], max_mag_diff=100):
        filtered = []
        has_segments = True
        if isinstance(data, Filtrate):
            has_segments = False
        for r in data:
            if has_segments:
                e1, e2, t, traveled_d, passing_d, segments, incidence_angle = r
            else:
                e1, e2, t, traveled_d, passing_d, totald, incidence_angle = r
            if abs(e1.magnitude-e2.magnitude)>max_mag_diff:
                continue

            if util.match_nslcs(ignore, [t.codes]):
                continue

            if traveled_d is False:
                continue

            gamma = traveled_d/passing_d
            if gamma<threshold_pass_factor or num.isnan(traveled_d) or traveled_d<min_travel_distance:
                continue
            else:
                filtered.append(r)
        if len(filtered)==0:
            raise Exception('len(filtered) == 0!')
        print '%s of %s pairs passed' %(len(filtered), len(data))
        return filtered

    def get_passing_distance(self, ray_points, x0):
        x1 = ray_points[:, :-1]
        x2 = ray_points[:, 1:]
        us = x2-x1
        us[:, num.where(us==0.)[1]] = num.nan
        vs = x0-x1
        ws = x0-x2
        u_norms = num.linalg.norm(us, axis=0)
        ps = num.sum(vs.T*us.T, axis=1) / u_norms**2

        # pp = passing point
        i_pp = num.where(num.logical_and(ps<1, ps>0))[0]
        if len(i_pp) == 0:
            # did not pass
            return False, False, []

        u = us[:, i_pp]
        p = ps[i_pp]
        projected_on_u = p * u
        traveled_distance = num.sum(u_norms[num.where(ps>1)])
        traveled_distance += num.linalg.norm(projected_on_u)

        # pp distance to segment
        d_pp = num.linalg.norm(num.cross(vs[:, i_pp], ws[:, i_pp], axis=0)) / u_norms[i_pp]
        segments = num.hstack((x1[:, :i_pp+1], x1[:, i_pp]+projected_on_u))
        return traveled_distance, d_pp, segments


    def get_passing_distance_old(self, ray_points, x0):
        passed = False
        passing_distance = False
        segments = [(ray_points[0], 0, passed)]
        traveled_distance = 0
        for i in xrange(len(ray_points)-1):
            x1, x2 = ray_points[i:i+2]
            u = x2-x1
            u_norm = num.linalg.norm(u)
            if u_norm == 0.:
                continue

            v = x0-x1
            w = x0-x2
            p = num.dot(v, u) / u_norm**2
            if p < 0 or p > 1:
                traveled_distance += u_norm
                seg = x2
            else:
                passing_distance = num.linalg.norm(num.cross(v, w))/u_norm
                projected_on_u = p * u
                seg = x1+projected_on_u
                traveled_distance += num.linalg.norm(projected_on_u)
                passed = True
            segments.append((seg, traveled_distance, passing_distance))
            if passed:
                break
        return passed, segments



class Animator():
    def __init__(self, pairs):
        self.pairs = pairs

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

    @staticmethod
    def plot_segments(self, segments, ax=None, color='r'):
        if not ax:
            fig, ax = get_3d_ax()
        for i, segment in enumerate(segments):
            (x,y,z), traveled_dist, passing_dist = segment
            ax.plot(x, y, z, c=color, alpha=0.1, linewidth=2)

    @staticmethod
    def get_3d_ax():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.invert_zaxis()
        #ax.set_aspect('equal')

        return fig, ax

    def make_animation(self, results, colormap):
        fig, ax = get_3d_ax()
        anim = animation.FuncAnimation(
            fig, animate, fargs=((ax, results, colormap)), frames=10, interval=15, blit=True)
        anim.save('coupled_rays.mp4', fps=1, extra_args=['-vcodec', 'libx264'])

    @staticmethod
    def plot_ray(segments, ax=None):
        if not ax:
            fig, ax = Animator.get_3d_ax()
        x, y, z = segments
        ax.plot(x, y, z )
        return ax

    @staticmethod
    def plot_sources(sources, reference=False, ax=None, alpha=1):
        if not ax:
            fig, ax = Animator.get_3d_ax()
        x, y, z = num.zeros(len(sources)), num.zeros(len(sources)), num.zeros(len(sources))
        if isinstance(sources, dict):
            srcs = sources.keys()
            count = sources.values()
        else:
            srcs = sources
            count = 1

        for i, s in enumerate(srcs):
            if reference:
                x[i], y[i], z[i] = reference(s)
            else:
                x[i], y[i], z[i] = (s.effective_latlon[0], s.effective_latlon[1], s.depth)
        ax.scatter(x, y, z, s=count, alpha=alpha)
        return ax

def array_center(stations):
    lats, lons, depths = [], [], []
    for s in stations:
        lats.append(s.lat)
        lons.append(s.lon)
        depths.append(s.depth)
    return gf.meta.Location(lat=float(num.mean(lats)),
                            lon=float(num.mean(lons)),
                            depth=float(num.mean(depths)))


def project2enz(arrival, azimuth_deg):
    azimuth = azimuth_deg*cake.d2r
    z, y, t = arrival.zxt_path_subdivided()
    e = num.sin(azimuth) * y[0]
    n = num.cos(azimuth) * y[0]
    return n, e, z[0]


def stats_by_station(results):
    hit_counter = {}
    for r in results:
        s1, s2, t, td, pd, totald, incidence_angle = r
        if not t in hit_counter:
            hit_counter[t] = 1
        else:
            hit_counter[t] += 1

    return hit_counter


def get_depths(results):
    depths = []
    for r in results:
        s1, s2, t, p, td= r
        depths.append(s1.depth)
        depths.append(s2.depth)
    return depths


def get_distances(results):
    distances = []
    for r in results:
        s1, s2, t, p, td= r
        distances.append(s1.distance_to(t))
        distances.append(s2.distance_to(t))

    return distances

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
    ax.set_ylabel('latitude  [$^\circ$]')
    fig.savefig('hitcount_map.png', dpi=200)
    plt.show()


def fresnel_lambda(total_length, td, pd):
    '''Wave length by Fresnel volume

    :param total_length: ray length
    :param td: traveled distance
    :param pd: passing distance
    '''
    d2 = total_length-td
    lda = pd**2*total_length/(td*d2)
    return lda


def scatter(X, Y, Z, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.scatter(X, Y, s=Z/num.max(Z)*50)
    return ax


def hitcount_map(hit_counter, stations, events=None, save_to='hitcount_map.png'):
    #data = num.zeros(len(hit_counter)*3).reshape(-1, 3)
    X, Y, Z = xyz_from_hitcount(hit_counter)
    ax = scatter(X, Y, Z)
    #check eins zwo
    #ax.set_xlim((min(X), max(X)))
    #ax.set_ylim((min(Y), max(Y)))
    plot_stations(stations, ax=ax)

    if events is not None:
        for e in events:
            ax.plot(e.lon, e.lat, 'ro', alpha=0.3)

    ax.set_ymargin(0.01)
    ax.set_xmargin(0.01)
    ax.set_xlabel('longitude [$^\circ$]')
    ax.set_ylabel('latitude [$^\circ$]')
    fig.savefig('hitcount_map.png', dpi=200)
    plt.tight_layout()



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


def plot_stations(stations, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for x, y, label in xyz_from_stations(webnet_stations):
        ax.plot(x, y, 'g^')
        ax.text(x, y, label,
                bbox={'facecolor':'white', 'alpha':0.5, 'pad':0.1, 'edgecolor':'white'})
    #for s in stations:
    #    ax.plot(s.lon, s.lat, 'to')
    return ax

def load_and_show(fn):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y, Z = num.loadtxt(fn).T
    grid_x, grid_y = num.mgrid[min(X):max(X):100j, min(Y):max(Y):100j]
    grid = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')
    ax.contourf(grid_x,grid_y, grid, cmap=cmap, levels=num.linspace(num.nanmin(grid), num.nanmax(grid), 30))
    ax = scatter(X, Y, Z, ax=ax)

if __name__=='__main__':
    year = 2008
    webnet_stations = model.load_stations('/data/meta/stations.pf')


    if True:
        print 'loading'
        load_and_show('hitcount_%s.txt' %year)
        plot_stations(webnet_stations, ax=plt.gca())
        plt.gcf().savefig('hitcount_map_%s.png' % year)
        plt.show()
        sys.exit(1)


    #events = list(model.Event.load_catalog('/data/meta/events2008.pf'))
    #compare_events = list(model.Event.load_catalog('/data/meta/events2008.pf'))
    #webnet_stations = model.load_stations('/data/meta/stations.pf')
    ##hitcount_map_from_file(filename='hitcount.txt', stations=webnet_stations)
    #stations = make_station_grid(webnet_stations, num_n=1, num_e=1, edge_stretch=0.15)
    #colormap = UniqueColor(tracers=stations)
    #phases = [cake.PhaseDef('p'), cake.PhaseDef('P')]
    #earthmodel = cake.load_model('../earthmodel_malek_alexandrakis.nd')

    ## sollte besser in 2d gemacht werden. Dauert ja sonst viel laenger...
    #fig, ax = Animator.get_3d_ax()

    #results = process(events, stations, earthmodel, phases)

    #filtered = filter_pairs(results, 5, 200., ax=ax)
    #hitcount = stats_by_station(filtered)
    #X, Y, Z = xyz_from_hitcount(hitcount)
    #num.savetxt('hitcount.txt', num.column_stack((X, Y, Z)))
    #make_animation(filtered, colormap)

    # here goes the later calculated full extension of the seismogenic zone
    #events = list(model.Event.load_catalog('/data/meta/events2008.pf'))

    #events = list(model.Event.load_catalog('/data/meta/webnet_reloc/hypo_dd_event.pf'))
    events = list(model.Event.load_catalog('/data/meta/webnet_reloc/reloc_new/%s_reloc.pf' %year))
    #compare_events = list(model.Event.load_catalog('/data/meta/events2008.pf'))
    #hitcount_map_from_file(filename='hitcount.txt', stations=webnet_stations)
    #print 'exiting.....'
    #stations = make_station_grid(webnet_stations, num_n=5, num_e=4, edge_stretch=0.2)
    stations = make_station_grid(webnet_stations, num_n=20, num_e=20, edge_stretch=1.0)
    plot_stations(stations)
    #colormap = UniqueColor(tracers=stations)
    #phases = [cake.PhaseDef('p'), cake.PhaseDef('P')]
    earthmodel = cake.load_model('models/earthmodel_malek_alexandrakis.nd')

    # sollte besser in 2d gemacht werden. Dauert ja sonst viel laenger...
    center = array_center(events)
    nevents = 80
    #nevents = 15
    coupler = Coupler()
    from q import s2t, e2s
    from pyrocko.gf import SourceWithMagnitude
    targets = [s2t(s) for s in stations]
    sources = [SourceWithMagnitude(lat=e.lat, lon=e.lon, depth=e.depth) for e in events]
    coupler.process(num.random.choice(sources, nevents), targets, earthmodel,
                    phases=None, ignore_segments=True)
    filtered = coupler.filter_pairs(4., 1000., data=coupler.filtrate)
    hitcount = stats_by_station(filtered)
    #hitcount_map(hitcount, webnet_stations, events)
    # save:
    X, Y, Z = xyz_from_hitcount(hitcount)
    num.savetxt('hitcount_%s.txt' % year, num.column_stack((X, Y, Z)))
    ax = scatter(X, Y, Z)
    grid_x, grid_y = num.mgrid[min(X):max(X):100j, min(Y):max(Y):100j]
    grid = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')
    ax.imshow(grid.T)

    #hitcount_map(filtered, webnet_stations)
    plt.show()
    sys.exit(0)
    distances = get_distances(filtered)
    depths = get_depths(filtered)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.hist(distances, bins=30)
    ax.set_title('epicentral distances')
    ax = fig.add_subplot(212)
    ax.hist(depths, bins=30)
    ax.set_title('source depths')
    fig.savefig('epi_dists_depths_pairs_newdd2008.png')
    #hitcount_pie(hitcount, colormap)
    #hitcount_map(hitcount, webnet_stations, events)
    plt.show()
