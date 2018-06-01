import numpy as num
import matplotlib.pyplot as plt
from scipy import signal

from pyrocko.pile import make_pile
from pyrocko.model import load_events, dump_events
from pyrocko.gui.marker import PhaseMarker, EventMarker, associate_phases_to_events
from pyrocko import trace
import mtspec


def do_demean(tr):
    tr.ydata = tr.ydata - num.mean(tr.ydata)


def do_normalize(tr, method='power'):
    y = tr.get_ydata()
    y = y / (num.sqrt(num.sum(y**2)) / (tr.tmax-tr.tmin))
    tr.set_ydata(y)


def cc_all(by_markers, hp, lp, tpad):

    def filt(tr):
        tr.highpass(4, hp)
        tr.lowpass(4, lp)

    def marker_to_name(m):
        return m.get_event().name

    markers = list(by_markers.keys())
    n_markers = len(markers)
    correlations = []
    for i_m, a_m in enumerate(markers):
        for j_m, b_m in enumerate(markers[i_m+1:]):
            a_tr = by_markers[a_m].copy()
            b_tr = by_markers[b_m].copy()
            do_demean(a_tr)
            do_demean(b_tr)
            filt(a_tr)
            filt(b_tr)
            # t, v = trace.correlate(a_tr, b_tr, normalization='normal', mode='full').max()
            t, v = trace.correlate(a_tr, b_tr, normalization='normal', mode='valid').max()
            correlations.append((marker_to_name(a_m), marker_to_name(b_m), v))
            # correlations.append((marker_to_name(b_m), marker_to_name(a_m), v))
    return correlations


def cc_batch(trs, hp, lp):
    ccs = []
    for itr_a, tr_a in enumerate(trs):
        ydata_a = tr_a.ydata

        tr_a = tr_a.copy()
        tr_a.highpass(4, hp)
        tr_a.lowpass(4, lp)
        for itr_b, tr_b in enumerate(trs[itr_a:]):
            ydata_b = tr_b.ydata
            tr_b = tr_b.copy()

            tr_b.highpass(4, hp)
            tr_b.lowpass(4, lp)
            t, v = trace.correlate(
                tr_a, tr_b, mode='full', normalization='normal').max()
            ccs.append(v)

    return num.mean(ccs)


class Scale():
    def __init__(self, values, key=None):
        self.vmin = min(values)
        self.vmax = max(values)
        self.key = key

    def __call__(self, val):
        if self.key:
            if self.key is None:
                raise Exception('Need to define a key')
            val = getattr(val, self.key)
        return (self.vmax - val) / (self.vmax-self.vmin)



def section_plot(by_markers, section_plot_key, filters, tpad, yscale_factor=0.5,
                 overlay=False, section_div=None):

    if section_div:
        def get_color(e):
            if getattr(e, section_plot_key) > section_div:
                return 'red'
            return 'blue'
    else:
        def get_color(e):
            return 'black'

    markers = by_markers.keys()
    events = [m.get_event() for m in markers]
    section_values = [float(getattr(item, section_plot_key)) for item in events]
    vmin = num.min(section_values)
    vmax = num.max(section_values)
    vrange = vmax - vmin

    fig, axs = plt.subplots(len(filters))
    for ifilt, (hp, lp) in enumerate(filters):
        ax = axs[ifilt]
        for marker, tr in by_markers.items():
            event = marker.get_event()
            ax.set_title('%s | %s - %s Hz' % (tr.station, hp, lp))
            tr = tr.copy()
            tr.highpass(4, hp)
            tr.lowpass(4, lp)
            # tr.chop(tr.tmin+tpad, tr.tmax-tpad)
            ydata = tr.get_ydata()
            ydata -= num.mean(ydata[0:10])
            ydata *= vrange * yscale_factor
            if not overlay:
                ydata = getattr(event, section_plot_key) + ydata
            ax.axvline(tr.tmin+tpad)
            ax.axvline(tr.tmax-tpad)
            ax.plot(
                tr.get_xdata(),
                ydata,
                color=get_color(event),
                alpha=0.08)


if __name__ == '__main__':
    # data_path = '/media/usb/vogtland/gse2'
    data_path = '/data/webnet/gse2/2008Oct'
    fn_markers = '/home/marius/josef_dd/hypodd_markers_josef.pf'
    fn_events = '/home/marius/josef_dd/events_from_sebastian.pf'
    fn_correlations = 'correlations'
    fn_index_mapping = 'index_mapping.txt'
    want_station = 'NKC'
    # want_station = 'LBC'
    want_channel = 'SHZ'
    normalize = True
    want_phase = 'P'
    section_plot_key = 'lat'
    section_div = 50.212
    magmin = 1.5
    demean = True
    yscale_factor = 0.3
    twin_min = -0.01
    twin_max = 0.2

    tpad = 0.5  # padding on both sides for filtering

    filters = [
        (10, 70),
        #(30, 70),
        # (50, 80),
        (1., 30),
        (6., 30),
    ]

    data_pile = make_pile(data_path, fileformat='gse2')
    markers = PhaseMarker.load_markers(fn_markers)
    events = load_events(fn_events)

    markers = [m for m in markers if m.one_nslc()[1] == want_station]
    markers = [m for m in markers if m.get_phasename().upper() == want_phase]
    event_markers = [EventMarker(e) for e in events]
    markers.extend(event_markers)
    associate_phases_to_events(markers)
    markers = [m for m in markers if isinstance(m, PhaseMarker)]
    markers = [m for m in markers if m.get_event().magnitude > magmin]
    events = list(set([m.get_event() for m in markers]))
    eventname_to_index = {e.name: i for i, e in enumerate(events)}
    eventname_to_event = {e.name: e for e in events}
    index_to_event = {i: eventname_to_event[en] for en, i in eventname_to_index.items()}

    markers.sort(key=lambda x: x.tmin)
    snippets = []

    batches = {
        lambda x: getattr(x, section_plot_key)>=section_div: [],
        lambda x: getattr(x, section_plot_key)<section_div: []
    }

    def get_batch(m):
        for k in batches.keys():
            if k(m):
                return batches[k]

    by_markers = {}
    markers.sort(key=lambda x: x.tmin)

    def iter_marker_with_snippets():
        for im, m in enumerate(markers):
            print('%s / %s' % (im+1, len(markers)))
            def selector(tr):
                return tr.channel == want_channel and tr.station == m.one_nslc()[1]
            for trs in data_pile.chopper(
                    trace_selector=selector,
                    tmin=m.tmin+twin_min-tpad,
                    tmax=m.tmin+twin_max+tpad,
                    keep_current_files_open=True):
                for tr in trs:
                    yield m, tr

    used_events = []
    for m, tr in iter_marker_with_snippets():
        event = m.get_event()
        used_events.append(m.get_event())
        batch = get_batch(event)
        tr.shift(-m.tmin)
        if demean:
            tr.ydata = tr.ydata - num.mean(tr.ydata)
        if normalize:
            do_normalize(tr)
        batch.append(tr)
        by_markers[m] = tr
    dump_events(used_events, fn_index_mapping.rsplit('.', 1)[0] + '_%s_use_events.pf'% want_station)

    fig, axs = plt.subplots(len(filters), len(batches))
    for ifilt, (hp, lp) in enumerate(filters):
        for ibatch, (event, batch) in enumerate(batches.items()):
            ax = axs[ifilt][ibatch]
            ax.set_title('%s | %s - %s Hz' % (want_station, hp, lp))
            for tr in batch:
                tr = tr.copy()
                tr.highpass(4, hp)
                tr.lowpass(4, lp)
                tr.chop(tr.tmin+tpad, tr.tmax-tpad)
                ax.plot(tr.get_xdata(), tr.get_ydata(), color='black', alpha=0.1)
            ax.text(0., 0., '%1.2f' % cc_batch(batch, hp, lp), transform=ax.transAxes)

    correlations = {}
    for filt in filters:
        with open(fn_correlations + '_seiscloud_%s_%s-%s.txt' %
                  (filt + (want_station, )), 'w') as f_seiscloud:
            with open(fn_correlations + '_%s_%s-%s.txt' % (filt + (want_station, )), 'w') as f:
                for (ma, mb, value) in cc_all(by_markers, *filt, tpad=tpad):
                    indexa = ma
                    indexb = mb
                    f.write('%s %s %1.4f\n' % (indexa, indexb, 1.-value))
                    indexa = eventname_to_index[ma]
                    indexb = eventname_to_index[mb]
                    f_seiscloud.write('%s %s %1.4f\n' % (indexa, indexb, 1.-value))

    with open(fn_index_mapping, 'w') as f:
        for vals in eventname_to_index.items():
            f.write('%s %s\n' % vals)

        indices = list(index_to_event.keys())
        indices.sort()
        catalog = []
        for i in indices:
            catalog.append(index_to_event[i])

        dump_events(catalog, fn_index_mapping.rsplit('.', 1)[0] + '_%s_cat.pf'% want_station)

    section_plot(by_markers, section_plot_key, filters, tpad=tpad, section_div=section_div, yscale_factor=yscale_factor)
    section_plot(by_markers, section_plot_key, filters, tpad=tpad,
                 section_div=section_div, yscale_factor=yscale_factor,
                 overlay=True)
    section_plot(by_markers, section_plot_key, filters, tpad=tpad, yscale_factor=yscale_factor)
    section_plot(by_markers, section_plot_key, filters, tpad=tpad,
                 yscale_factor=yscale_factor, overlay=True)

    plt.show()

