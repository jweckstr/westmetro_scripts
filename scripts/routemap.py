import os

import numpy as np
import pandas as pd
import matplotlib.mlab as ml
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx
import matplotlib.pyplot as plt
import shapely.geometry.linestring as shapely
from shapely.geometry.point import Point
from geopandas import GeoDataFrame, sjoin

from gtfspy.routing.journey_data import JourneyDataManager, DiffDataManager
from gtfspy.routing.journey_data_analyzer import JourneyDataAnalyzer
from research.westmetro_paper.scripts.all_to_all_settings import *
from research.westmetro_paper.scripts.all_to_all_analyzer import AllToAllDifferenceAnalyzer
import gtfspy.smopy_plot_helper

from gtfspy.mapviz import *
from gtfspy.colormaps import *
from gtfspy.util import makedirs
from gtfspy.gtfs import GTFS
from gtfspy.shapes import get_shape_between_stops
from gtfspy.route_types import *
from gtfspy.stats import get_section_stats
from gtfspy.networks import route_to_route_network
"""
1. SELECT the longest trip (in time) for each route that is active at the desired time
2. cluster stops by name and location
3. split shapes by stop section
4. select representative shape for each stop-to-stop section
5. deal with skip-stop service
6. route ranking for ordering, this could be done by looping trough all route sections route for route->and assign the
route code for each stop. direction is determined by the sections stop id's, assuming smaller to larger. 
if the case is the opposite, the new routes will be added to the left in the queue
7. offset routes

"""
if False:
    gtfs_name = "lm_daily"
else:
    gtfs_name = "old_daily"

gtfs = GTFS(FEED_DICT[gtfs_name]["gtfs_dir"])
bunching_value = 9
line_spacing = 0.0001
shapes = False


def get_linestrings_for_stop_section(stop_tuple, trip_id, from_shape_brake, to_shape_brake):
    try:
        assert shapes
        shapedict = get_shape_between_stops(gtfs.conn.cursor(),
                                            trip_id,
                                            stop_tuple[0],
                                            stop_tuple[1],
                                            (from_shape_brake, to_shape_brake))
        assert not len(set(shapedict["lat"])) <= 1
        assert not len(set(shapedict["lon"])) <= 1
        return shapely.LineString([(lon, lat) for lat, lon in zip(shapedict["lat"], shapedict["lon"])])
    except (ValueError, AssertionError):
        lat0, lon0 = gtfs.get_stop_coordinates(stop_tuple[0])
        lat1, lon1 = gtfs.get_stop_coordinates(stop_tuple[1])
        if lat0 == lat1 and lon0 == lon1:
            return
        else:
            return shapely.LineString([(lon0, lat0), (lon1, lat1)])


def route_parallels(line, route, all_routes, bunching_value=bunching_value, line_spacing=line_spacing):
    n_parallels = len(all_routes)
    line_routes = []
    if not line:
        return

    if n_parallels < bunching_value:
        offsets = np.linspace(-1 * ((n_parallels - 1) * line_spacing) / 2,
                              ((n_parallels - 1) * line_spacing) / 2, n_parallels)
        try:
            return line.parallel_offset(abs(offsets[all_routes.index(route)]), "left" if offsets[all_routes.index(route)] < 0 else "right")
        except:
            print(line, offsets[all_routes.index(route)])
    else:
        return line


def get_route_ranking(df):
    route_order_for_stop_sections = {}
    stop_section_shapes = {}
    for row in df.itertuples():
        section_tuple = (row.from_stop_I, row.to_stop_I)
        alt_section_tuple = (row.to_stop_I, row.from_stop_I)
        if not section_tuple in route_order_for_stop_sections and not alt_section_tuple in route_order_for_stop_sections:
            route_order_for_stop_sections[section_tuple] = [row.route_I]
            stop_section_shapes[section_tuple] = (row.trip_I, row.from_shape_break, row.to_shape_break)
        elif section_tuple in route_order_for_stop_sections:
            route_order_for_stop_sections[section_tuple].append(row.route_I)
        elif alt_section_tuple in route_order_for_stop_sections:
            route_order_for_stop_sections[alt_section_tuple].insert(0, row.route_I)
    return route_order_for_stop_sections, stop_section_shapes

def get_geometry(stop_tuple, route, all_routes, cluster_dict):
    #line = get_linestrings_for_stop_section(stop_tuple, trip_id, from_shape_break, to_shape_break)
    #print(stop_tuple, cluster_dict[stop_tuple[0]], cluster_dict[stop_tuple[1]])
    line = shapely.LineString([cluster_dict[stop_tuple[0]][0], cluster_dict[stop_tuple[1]][0]])
    if stop_tuple[0] == stop_tuple[1]:
        return
    else:
        return route_parallels(line, route, all_routes, bunching_value=bunching_value, line_spacing=line_spacing)


def cluster_stops(stops_set):
    df = gtfs.execute_custom_query_pandas("""SELECT * FROM stops
                                             WHERE stop_I IN ({stops_set})""".format(stops_set=",".join([str(x) for x in stops_set])))
    df["geometry"] = df.apply(lambda row: Point((row["lon"], row["lat"])), axis=1)
    crs_wgs = {'init': 'epsg:4326'}
    crs_eurefin = {'init': 'epsg:3067'}
    gdf = GeoDataFrame(df, crs=crs_wgs, geometry=df["geometry"])
    gdf = gdf.to_crs(crs_eurefin)
    gdf_poly = gdf.copy()
    gdf_poly["geometry"] = gdf_poly["geometry"].buffer(75)
    gdf_poly["everything"] = 1

    gdf_poly = gdf_poly.dissolve(by="everything")

    polygons = None
    for geoms in gdf_poly["geometry"]:
        polygons = [polygon for polygon in geoms]

    single_parts = GeoDataFrame(crs=crs_eurefin, geometry=polygons)
    single_parts['new_stop_I'] = single_parts.index
    gdf_joined = sjoin(gdf, single_parts, how="left", op='within')
    single_parts["geometry"] = single_parts.centroid
    gdf_joined = gdf_joined.drop('geometry', 1)
    centroid_stops = single_parts.merge(gdf_joined, on="new_stop_I")
    return centroid_stops.to_crs(crs_wgs)
    #fig, ax = plt.subplots()
    #centroid_stops.plot(cmap='viridis')
    #plt.show()

    """
    change projection for accurate buffer distance
    merge polygons,
    select single parts
    calculate centroids
    """


def main():
    """

    :return:
    """
    query = """
            WITH 
                a AS (
                SELECT routes.name AS name, shape_id, route_I, trip_I, routes.type, max(end_time_ds-start_time_ds) AS trip_duration, count(*) AS n_trips
                FROM trips
                LEFT JOIN routes 
                USING(route_I)
                WHERE start_time_ds >= 7*3600 AND start_time_ds < 8*3600
                GROUP BY routes.route_I
                ),
                b AS(
                SELECT q1.trip_I AS trip_I, q1.stop_I AS from_stop_I, q2.stop_I AS to_stop_I, q1.seq AS seq, q1.shape_break AS from_shape_break, q2.shape_break AS to_shape_break FROM
                (SELECT stop_I, trip_I, shape_break, seq FROM stop_times) q1,
                (SELECT stop_I, trip_I, shape_break, seq AS seq FROM stop_times) q2
                WHERE q1.seq=q2.seq-1 AND q1.trip_I=q2.trip_I AND q1.trip_I IN (SELECT trip_I FROM a)
                )
            SELECT b.*, name, route_I, shape_id FROM b, a
            WHERE b.trip_I = a.trip_I
            ORDER BY route_I, seq
            """
    df = gtfs.execute_custom_query_pandas(query)
    stops_set = set(df["from_stop_I"]) | set(df["to_stop_I"])
    clustered_stops = cluster_stops(stops_set)
    cluster_dict = clustered_stops[["new_stop_I", "stop_I", "geometry"]].set_index('stop_I').T.to_dict('list')
    geom_dict = clustered_stops[["new_stop_I", "geometry"]].set_index("new_stop_I").T.to_dict('list')
    df["to_stop_I"] = df.apply(lambda row: cluster_dict[row["to_stop_I"]][0], axis=1)
    df["from_stop_I"] = df.apply(lambda row: cluster_dict[row["from_stop_I"]][0], axis=1)

    route_order_for_stop_sections, stop_section_shapes = get_route_ranking(df)

    df["section_tuple"] = df.apply(lambda row: (row['from_stop_I'], row['to_stop_I'])
                                   if (row['from_stop_I'], row['to_stop_I']) in route_order_for_stop_sections.keys()
                                   else (row['to_stop_I'], row['from_stop_I']), axis=1)
    #df["master_trip_I"] = df.apply(lambda row: stop_section_shapes[row["section_tuple"]][0], axis=1)
    df["routes_on_section"] = df.apply(lambda row: route_order_for_stop_sections[row["section_tuple"]], axis=1)
    df["geometry"] = df.apply(lambda row: get_geometry(row["section_tuple"], row["route_I"], row["routes_on_section"],
                                                       geom_dict),
                              axis=1)
    df["n_routes"] = df.apply(lambda row: len(row['routes_on_section']), axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df, crs=crs, geometry=df["geometry"])
    gdf = gdf[pd.notnull(gdf['geometry'])]
    gdf = gdf.drop(['section_tuple', "routes_on_section"], 1)
    #gdf.to_file(driver='ESRI Shapefile', filename=os.path.join("/home/clepe/production/results/helsinki/figs/transit_maps/", gtfs_name + "transit.result.shp"))
    print("plotting...")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="smopy_axes")
    #gdf = gdf.loc[gdf['section_tuple'] == (2193, 2217)]
    #gdf.plot(column="route_I", cmap='prism')
    #zoom = 2
    #w, h = fig.get_size_inches()

    #fig.tight_layout()
    #fig.set_size_inches(w * zoom, h * zoom)
    cmap = matplotlib.cm.get_cmap(name='prism', lut=None)
    ax.set_map_bounds(**SPATIAL_BOUNDS)
    for ix, row in gdf.iterrows():
        color = "r" if row["route_I"] % 2 == 0 else "b"
        ax.plot_geometry(row["geometry"], update=False, color=color)
    plt.show()
    #fig.savefig(os.path.join("/home/clepe/production/results/helsinki/figs/transit_maps/", gtfs_name + "transit.png"),
     #           format="png", dpi=300)


main()
