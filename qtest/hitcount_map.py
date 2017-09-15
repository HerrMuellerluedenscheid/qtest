import numpy as num
from pyrocko import model
from pyrocko.plot.automap import Map
#from pyrocko.automap import Map


hitcount_fn = 'hitcount_2008.txt'
stations = model.load_stations('/data/meta/stations.pf')


# Generate the basic map
m = Map(
    lat=stations[0].lat,
    lon=stations[0].lon,
    radius=40000.,
    width=30., height=30.,
    show_grid=False,
    show_topo=True,
    color_dry=(238, 236, 230),
    topo_cpt_wet='light_sea_uniform',
    topo_cpt_dry='light_land_uniform',
    illuminate=True,
    illuminate_factor_ocean=0.15,
    show_rivers=False,
    show_plates=True)

# Draw some larger cities covered by the map area
m.draw_cities()

# Generate with latitute, longitude and labels of the stations
lats = [s.lat for s in stations]
lons = [s.lon for s in stations]
labels = ['.'.join(s.nsl()) for s in stations]

# Stations as black triangles. Genuine GMT commands can be parsed by the maps'
# gmt attribute. Last argument of the psxy function call pipes the maps'
# pojection system.
m.gmt.psxy(in_columns=(lons, lats), S='t20p', G='black', *m.jxyr)

# Station labels
for i in range(len(stations)):
    m.add_label(lats[i], lons[i], labels[i])

d = num.loadtxt(hitcount_fn)
psxy_data = []
for lon, lat, n in d:
    print(lon, lat)
    psxy_data.append((lon, lat, n, 3))

# Draw a beachball
m.gmt.psxy(S='m.5', G='red', in_rows=psxy_data, *m.jxyr)

m.save('automap_vogtland.png')
