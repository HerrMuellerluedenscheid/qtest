import numpy as num
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from pyrocko import cake, gf, model
from pyrocko import orthodrome as ortho
from pyrocko import util
from pyrocko.gf import Target, meta, SourceWithMagnitude
from pyrocko.guts import List, Object, String, Float
import logging
import sys
import os
import hashlib
try:
    import cPickle as pickle
except ImportError:
    import pickle

from .invert import Ray3D, Ray3DDiscretized
from .vtk_graph import vtk_ray, render_actors, vtk_point


logger = logging.getLogger()

cmap = plt.cm.bone_r
pjoin = os.path.join


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
    magdiffmax = Float.T(optional=True)
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

    def dump_pickle(self, filename):
        pickle.dump(self, open(filename, 'wb'))
        logger.info('dump cpickle: %s' % filename)

    @classmethod
    def load_pickle(cls, filename):
        logger.info('load cpickle: %s' % filename)
        return pickle.load(open(filename, 'rb'))


def filename_hash(sources, targets, earthmodel, phases):
    hstr = ''
    for s in sources:
        hstr += str(s)
    for t in targets:
        hstr += str(t)
    hstr += str(earthmodel)
    for phase in phases:
        hstr += str(phase)

    return hashlib.sha1(hstr.encode('utf-8')).hexdigest()


class Coupler():
    def __init__(self, filtrate=None):
        self.results = []
        self.hookup = None
        self.filtrate = filtrate
        self.magdiffmax = 100.
        self.minimum_magnitude = -10

    def process(self, sources, targets, earthmodel, phases, ignore_segments=True,
                fn_cache=None, check_relevance_by=False):

        self.filtrate = Filtrate(
            sources=sources, targets=targets, phases=phases,
            earthmodel=earthmodel, magdiffmax=self.magdiffmax)

        i = 0
        self.hookup = Hookup.from_locations(targets)
        failed = 0
        passed = 0
        nsources = len(sources)
        logger.info('start coupling events')
        actors = []
        for i_e, ev in enumerate(sources):
            #actors.append(vtk_point(self.hookup(ev)))
            for i_t, t in enumerate(targets):
                if not self.is_relevant(ev, t, check_relevance_by):
                    continue
                arrivals = earthmodel.arrivals([t.distance_to(ev)*cake.m2d], phases=phases, zstart=ev.depth) 

                if len(arrivals) == 0:
                    print('no arrival', t, ev.depth, phases)
                    continue
                # first arrivals
                arrival = arrivals[0]
                incidence_angle = arrival.incidence_angle()
                try:
                    n, e, d, travel_times = project2enz(arrival, ev.azibazi_to(t)[0])
                except IndexError as err:
                    print('!!!Error> ', err)
                    continue

                n = n * cake.d2m
                e = e * cake.d2m
                #if False:
                    # old version
                points_of_segments = self.hookup.add_to_ned(ev, (n,e,d))

                #else:
                #    r = Ray3D.from_RayPath(arrival[0])
                #    r.set_carthesian_coordinates(*self.hookup(ev))
                #    #r.set_carthesian_coordinates(n, e, d)
                #    points_of_segments = num.array(r.orientate3d()[:3])

                #actors.append(vtk_ray(points_of_segments, opacity=0.3))
                for i_cmp_e, cmp_e in enumerate(sources):
                    # print('y', ev.name, cmp_e.name)
                    ned_cmp = self.hookup(cmp_e)
                    #if abs(ev.magnitude-cmp_e.magnitude)>self.filtrate.magdiffmax:
                    #    failed += 1
                    #    continue

                    aout = get_passing_distance(points_of_segments, num.array(ned_cmp))
                    if aout:
                        td, pd, total_td, sgmts = aout
                        travel_time_segment = travel_times[0][:sgmts.shape[1]][-1]
                        # print(travel_time_segment)
                        # print(aout, travel_time_segment)
                        # print(travel_time_segment)
                        # ray = Ray3DDiscretized(*sgmts, t=travel_times[0][:sgmts.shape[1]])
                        #actors.append(vtk_ray(num.array(ray.nezt[0:3]), opacity=0.3))
                        passed += 1
                        self.filtrate.couples.append((
                            ev, cmp_e, t,
                            float(td), float(pd), float(total_td),
                            incidence_angle, travel_time_segment
                            ))
                    continue

            i += 1
            print('%s / %s' % (i, nsources))
        # VTK
        # render_actors(actors)
        print('failed: %s, passed:%s ' % (failed, passed))
        if passed == 0:
            raise Exception('Coupling failed')

        if fn_cache:
            self.filtrate.dump_pickle(filename=fn_cache)

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

    def filter_pairs(self, threshold_pass_factor, min_travel_distance, data,
                     ignore=[], max_mag_diff=100, max_magnitude=10.,
                     min_magnitude=-10.):

        filtered = []
        has_segments = True
        #if isinstance(data, Filtrate):
        #has_segments = False

        for r in data:
            e1, e2, t, traveled_d, passing_d, totald, incidence_angle, travel_time_segment = r
            #e1, e2, t, traveled_d, passing_d, segments, totald, incidence_angle = r

            if e1.magnitude > max_magnitude or e2.magnitude> max_magnitude:
                continue

            if e1.magnitude < min_magnitude or e2.magnitude< min_magnitude:
                continue

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
        print('%s of %s pairs passed' %(len(filtered), len(data)))
        return filtered

def get_passing_distance(ray_points, x0):
    us = ray_points[:, 1:] - ray_points[:, :-1]

    vs = x0 - ray_points[:, :-1]
    ws = x0 - ray_points[:, 1:]

    #vs = x0-x1
    #ws = x0-x2
    u_norms = num.linalg.norm(us, axis=0)
    ps = num.sum(vs*us, axis=0) / u_norms**2

    # pp = passing point
    i_pp = num.where(num.logical_and(ps<1, ps>0))[0]

    if len(i_pp) != 1:
        # did not pass
        return None

    i_pp = i_pp[0]
    projected_on_u = ps[i_pp] * us[:, i_pp]
    total_traveled_distance = num.nansum(u_norms)
    traveled_distance = num.sum(u_norms[:i_pp]) + num.linalg.norm(projected_on_u)

    # pp distance to segment
    d_pp = num.linalg.norm(num.cross(vs[:, i_pp], ws[:, i_pp], axis=0)) / u_norms[i_pp]
    segments = num.hstack(
        (ray_points[:, :i_pp+1], (ray_points[:, i_pp+1] -
                                projected_on_u).reshape(3,1)))
    #traveled_distance = num.sum(num.linalg.norm(segments.T[1:]-segments.T[:-1], axis=0))
    #print traveled_distance
    #r1 = vtk_ray(segments)
    #r2 = vtk_point(x0)
    #vtk_render([r1, r2])
    return traveled_distance, d_pp, total_traveled_distance, segments



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
    z, y, t = arrival.zxt_path_subdivided(points_per_straight=400)
    # z, y, t = arrival.zxt_path_subdivided(points_per_straight=800)
    e = num.sin(azimuth) * y[0]
    n = num.cos(azimuth) * y[0]
    return n, e, z[0], t


def stats_by_station(results):
    hit_counter = {}
    for r in results:
        print(r)
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
    from util import s2t, e2s
    from pyrocko.gf import SourceWithMagnitude

    year = 2008
    webnet_stations = model.load_stations('/data/meta/stations.pf')

    if False:
        print('loading')
        load_and_show('hitcount_%s.txt' %year)
        plot_stations(webnet_stations, ax=plt.gca())
        plt.gcf().savefig('hitcount_map_%s.png' % year)
        plt.show()
        sys.exit(1)

    eventfn = '/home/marius/josef_dd/events_from_sebastian_check_M1.pf'
    events = list(model.Event.load_catalog(eventfn))
    compare_events = list(model.Event.load_catalog(eventfn))
    phases = [cake.PhaseDef('p'), cake.PhaseDef('P')]
    earthmodel = cake.load_model('/data/models/earthmodel_malek_alexandrakis.nd')
    nevents = 80
    coupler = Coupler()
    stations = make_station_grid(webnet_stations, num_n=25, num_e=25,
                                 edge_stretch=1.75)
    targets = [s2t(s) for s in stations]
    sources = [SourceWithMagnitude(lat=e.lat, lon=e.lon, depth=e.depth) for e in events]
    coupler.process(num.random.choice(sources, nevents), targets, earthmodel,
                    phases=phases, ignore_segments=True)
    filtered = coupler.filter_pairs(3., 1000., data=coupler.filtrate)
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
    #plt.show()
    #sys.exit(0)
    #distances = get_distances(filtered)
    #depths = get_depths(filtered)
    #fig = plt.figure()
    #ax = fig.add_subplot(211)
    #ax.hist(distances, bins=30)
    #ax.set_title('epicentral distances')
    #ax = fig.add_subplot(212)
    #ax.hist(depths, bins=30)
    #ax.set_title('source depths')
    #fig.savefig('epi_dists_depths_pairs_newdd2008.png')
    #hitcount_pie(hitcount, colormap)
    #hitcount_map(hitcount, webnet_stations, events)
    plt.show()
