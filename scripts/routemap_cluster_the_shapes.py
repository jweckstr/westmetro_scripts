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
from shapely.ops import split
from shapely.wkt import loads
import copy
from geopandas import GeoDataFrame, sjoin

from gtfspy.routing.journey_data import JourneyDataManager, DiffDataManager
from gtfspy.routing.journey_data_analyzer import JourneyDataAnalyzer
from scripts.all_to_all_settings import *
from scripts.all_to_all_analyzer import AllToAllDifferenceAnalyzer

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

if True:
    gtfs_name = "lm_daily"
else:
    gtfs_name = "old_daily"


class RouteMapMaker:
    def __init__(self, gtfs_name):
        if isinstance(gtfs_name, str):
            self.gtfs = GTFS(FEED_DICT[gtfs_name]["gtfs_dir"])
        else:
            self.gtfs = gtfs_name
        self.bunching_value = 99
        self.line_spacing = 0.0001
        self.shapes = False
        self.crs_wgs = {'init': 'epsg:4326'}
        #self.crs_eurefin = {'init': 'epsg:3067'}

    def cluster_shapes(self):
        """

        :return:
        """
        # get unique stop-to-stop shapes, with trips aggregated
        # split by nearby stops
        # match splitted, and aggregate trips
        # identify branches: large overlap but crosses buffer, insert pseudo stop at branch
        # split everything again
        # match splitted

        #this query returns shapes for of the maximum trips, both directions
        df = self.gtfs.execute_custom_query_pandas(
             """WITH 
                a AS (
                SELECT routes.name AS name, shape_id, route_I, trip_I, routes.type, direction_id, 
                max(end_time_ds-start_time_ds) AS trip_duration, count(*) AS n_trips
                FROM trips
                LEFT JOIN routes 
                USING(route_I)
                WHERE start_time_ds >= 7*3600 AND start_time_ds < 8*3600
                GROUP BY routes.route_I, direction_id
                ),
                b AS(
                SELECT q1.trip_I AS trip_I, q1.stop_I AS from_stop_I, q2.stop_I AS to_stop_I, q1.seq AS seq, 
                q1.shape_break AS from_shape_break, q2.shape_break AS to_shape_break FROM
                (SELECT stop_I, trip_I, shape_break, seq FROM stop_times) q1,
                (SELECT stop_I, trip_I, shape_break, seq AS seq FROM stop_times) q2
                WHERE q1.seq=q2.seq-1 AND q1.trip_I=q2.trip_I AND q1.trip_I IN (SELECT trip_I FROM a)
                ),
                c AS(
                SELECT b.*, name, direction_id, route_I, a.shape_id, group_concat(lat) AS lats, 
                group_concat(lon) AS lons, count(*) AS n_coords FROM b, a, shapes
                WHERE b.trip_I = a.trip_I AND shapes.shape_id=a.shape_id
                AND b.from_shape_break <= shapes.seq AND b.to_shape_break >= shapes.seq
                GROUP BY route_I, direction_id, b.seq
                ORDER BY route_I, b.seq
                )
                SELECT from_stop_I, to_stop_I, group_concat(trip_I) AS trip_ids, 
                group_concat(direction_id) AS direction_ids, lats, lons FROM c
                WHERE n_coords > 1
                GROUP BY from_stop_I, to_stop_I
                ORDER BY count(*) DESC""")

        df["geometry"] = df.apply(lambda row:
                                  shapely.LineString([(float(lon), float(lat)) for lon, lat in
                                                      zip(row["lons"].split(","), row["lats"].split(","))]), axis=1)

        gdf = GeoDataFrame(df, crs=self.crs_wgs, geometry=df["geometry"])
        #gdf = gdf.to_crs(self.crs_eurefin)
        gdf = gdf.to_crs(self.crs_wgs)

        gdf = gdf.drop(["lats", "lons"], axis=1)

        stops_set = set(gdf["from_stop_I"]) | set(gdf["to_stop_I"])
        gdf["orig_parent_stops"] = list(zip(gdf['from_stop_I'], gdf['to_stop_I']))
        clustered_stops = self.cluster_stops(stops_set)
        cluster_dict = clustered_stops[["new_stop_I", "stop_I", "geometry"]].set_index('stop_I').T.to_dict('list')
        geom_dict = clustered_stops[["new_stop_I", "geometry"]].set_index("new_stop_I").T.to_dict('list')
        gdf["to_stop_I"] = gdf.apply(lambda row: cluster_dict[row["to_stop_I"]][0], axis=1)
        gdf["from_stop_I"] = gdf.apply(lambda row: cluster_dict[row["from_stop_I"]][0], axis=1)
        # to/from_stop_I: cluster id
        # orig_parent_stops: old id
        # child_stop_I: cluster id
        splitted_gdf = self.split_shapes_by_nearby_stops(clustered_stops, gdf)
        splitted_gdf['child_stop_I'] = splitted_gdf.apply(lambda row: ",".join([str(int(x)) for x in row.child_stop_I]), axis=1)
        splitted_gdf_grouped = splitted_gdf.groupby(['child_stop_I'])
        splitted_gdf_grouped = splitted_gdf_grouped.agg({'orig_parent_stops': lambda x: tuple(x),
                                                         'geometry': lambda x: x.iloc[0]}, axis=1)

        splitted_gdf = splitted_gdf_grouped.reset_index()
        splitted_gdf['value'] = splitted_gdf.apply(lambda row: 1, axis=1)
        #splitted_gdf = splitted_gdf.set_geometry(splitted_gdf["geometry"], crs=self.crs_eurefin)

        splitted_gdf = self.match_shapes(splitted_gdf)
        splitted_gdf["rand"] = np.random.randint(1, 10, splitted_gdf.shape[0])
        print(splitted_gdf)
        self.plot_geopandas(splitted_gdf, alpha=0.3)

    def split_shapes_by_nearby_stops(self, stops, shapes, buffer=0.01):
        """
        Splits shapes by stops, within buffer
        :param stops: GeoDataFrame
        :param shapes:
        :return:
        """
        # stops within buffer
        # splitter
        # retain the "parent" stop section
        #stops['geometry'] = stops.apply(lambda row: str(row.geometry), axis=1)

        #stops = stops.groupby(['new_stop_I', 'geometry'])['stop_I'].apply(list).reset_index()
        #stops["geometry"] = stops.apply(lambda row: loads(row.geometry), axis=1)

        stops_grouped = stops.groupby(['new_stop_I'])
        stops_grouped = stops_grouped.agg({'stop_I': lambda x: tuple(x),
                                           'geometry': lambda x: x.iloc[0]}, axis=1)

        stops = stops_grouped.reset_index()

        #stops = stops.set_geometry(stops["geometry"], crs=self.crs_eurefin)
        stops["point_geom"] = stops["geometry"]

        shapes["buffer"] = shapes["geometry"].buffer(buffer)
        shapes["line_geom"] = shapes["geometry"]
        shapes = shapes.set_geometry(shapes["buffer"])

        gdf_joined = sjoin(shapes, stops, how="left", op='intersects')
        gdf_joined = gdf_joined.set_geometry(gdf_joined["line_geom"])
        gdf_joined = gdf_joined.drop(["buffer", "line_geom"], axis=1)
        #gdf_joined['geometry'] = gdf_joined.apply(lambda row: str(row.geometry), axis=1)

        gdf_grouped = gdf_joined.groupby(["orig_parent_stops", 'from_stop_I', 'to_stop_I'])
        gdf_grouped = gdf_grouped.agg({'point_geom': lambda x: tuple(x),
                                       'new_stop_I': lambda x: tuple(x),
                                       'geometry': lambda x: x.iloc[0]}, axis=1)
        gdf_joined = gdf_grouped.reset_index()

        gdf_joined = gdf_joined.apply(lambda row: self.split_shape_by_points(row), axis=1)

        new_list = []
        for row in gdf_joined.to_dict('records'):
            for shape, stop_tuple in zip(row['shape_parts'], row['child_stop_Is']):
                new_row = copy.deepcopy(row)
                new_row["shape_part"] = shape
                new_row["child_stop_I"] = stop_tuple
                new_list.append(new_row)
        gdf_joined = pd.DataFrame(new_list)
        gdf_joined = gdf_joined.set_geometry(gdf_joined["shape_part"])
        return gdf_joined[['child_stop_I', 'orig_parent_stops', 'geometry']]

    def check_shape_orientation(self, shape, from_stop_point, to_stop_point):
        """
        Checks that the shape goes from the from stop to the to stop and not the opposite direction
        :param shape:
        :param from_stop_point:
        :param to_stop_point:
        :return:
        """

#    def split_shape_by_points(self, shape, shape_parents, points, point_ids):

    def split_shape_by_points(self, row):
        """

        :param shape:
        :param shape_parents:
        :param points:
        :param point_ids:
        :return:
        """
        shape = row["geometry"]
        shape_parents = [row["from_stop_I"], row["to_stop_I"]]
        points = row["point_geom"]
        point_ids = row["new_stop_I"]
        # TODO: change this to also output the cluster point ids for the end points so that matching is possible directly
        if not isinstance(points[0], shapely.Point):
            row["shape_parts"] = [shape]
            row["child_stop_Is"] = [shape_parents]
            return row
        # finds the distance on the shape that corresponds to the closest distance to the point
        distance_dict = {shape.project(point): {"point": point, "id": id} for point, id in zip(points, point_ids)}
        shape_parts = []
        stop_sections = []
        rest_of_shape = copy.deepcopy(shape)
        previous_stop = shape_parents[0]
        # loops trough the points in the order they are compared to the shape
        if len(distance_dict) >= 3:
            for key in sorted(distance_dict)[1:-1]:
                if distance_dict[key]["id"] not in shape_parents:
                    new_point = shape.interpolate(key)

                    # TODO: this step only works with a modified version of split(), replace with a custom function
                    geometries = split(rest_of_shape, new_point)

                    stop_sections.append((int(previous_stop), int(distance_dict[key]["id"])))
                    previous_stop = distance_dict[key]["id"]

                    if len(geometries) == 2:
                        rest_of_shape = geometries[1]
                        shape_parts.append(geometries[0])
                    else:
                        rest_of_shape = geometries[0]

        shape_parts.append(rest_of_shape)
        stop_sections.append((previous_stop, shape_parents[1]))
        #if len(shape_parts) > 1:
        #    assert not all(x == shape_parts[0] for x in shape_parts)

        #shape_parts = row["new_stop_I"]
        #stop_sections = row["new_stop_I"]
        row["shape_parts"] = shape_parts
        row["child_stop_Is"] = stop_sections
        return row
#        return (shape_parts, stop_sections)

    def match_shapes(self, shapes, buffer=0.01):
        """
        checks if shapes are completely within each others buffers, aggregates routes for these
        :return:
        """
        # buffer for spatial self join
        first_points = shapes["geometry"].apply(lambda x: Point(x.coords[0]))
        last_points = shapes["geometry"].apply(lambda x: Point(x.coords[-1]))

        points = pd.concat([first_points, last_points])
        point_df = points.to_frame(name='geometry')
        #point_df = point_df.set_geometry(point_df["geometry"], crs=self.crs_eurefin)
        point_df = point_df.set_geometry(point_df["geometry"], crs=self.crs_wgs)

        #buffer = point_df.buffer(30)
        #buffer = GeoDataFrame(crs=self.crs_eurefin, geometry=point_df.buffer(buffer))
        buffer = GeoDataFrame(crs=self.crs_wgs, geometry=point_df.buffer(buffer))

        buffer["everything"] = 1
        gdf_poly = buffer.dissolve(by="everything")
        polygons = None
        for geoms in gdf_poly["geometry"]:
            polygons = [polygon for polygon in geoms]

        #single_parts = GeoDataFrame(crs=self.crs_eurefin, geometry=polygons)
        single_parts = GeoDataFrame(crs=self.crs_wgs, geometry=polygons)

        single_parts['new_stop_I'] = single_parts.index

        gdf_joined = sjoin(shapes, single_parts, how="left", op='within')
        return gdf_joined

    def identify_branches(self, shapes, buffer=0.01):
        """
        Checks for other shapes that exits the buffer of another buffer. In these cases a pseudo stop is created,
        for further splitting of shapes
        :param shapes:
        :param buffer:
        :return:
        """

    def get_linestrings_for_stop_section(self, stop_tuple, trip_id, from_shape_brake, to_shape_brake):
        try:
            assert self.shapes
            shapedict = get_shape_between_stops(self.gtfs.conn.cursor(),
                                                trip_id,
                                                stop_tuple[0],
                                                stop_tuple[1],
                                                (from_shape_brake, to_shape_brake))
            assert not len(set(shapedict["lat"])) <= 1
            assert not len(set(shapedict["lon"])) <= 1
            return shapely.LineString([(lon, lat) for lat, lon in zip(shapedict["lat"], shapedict["lon"])])
        except (ValueError, AssertionError):
            lat0, lon0 = self.gtfs.get_stop_coordinates(stop_tuple[0])
            lat1, lon1 = self.gtfs.get_stop_coordinates(stop_tuple[1])
            if lat0 == lat1 and lon0 == lon1:
                return
            else:
                return shapely.LineString([(lon0, lat0), (lon1, lat1)])

    def route_parallels(self, line, route, all_routes, bunching_value=5, line_spacing=0.0001):
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

    def get_route_ranking(self, df):
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

    def get_geometry(self, stop_tuple, route, all_routes, cluster_dict):
        #line = get_linestrings_for_stop_section(stop_tuple, trip_id, from_shape_break, to_shape_break)
        #print(stop_tuple, cluster_dict[stop_tuple[0]], cluster_dict[stop_tuple[1]])
        line = shapely.LineString([cluster_dict[stop_tuple[0]][0], cluster_dict[stop_tuple[1]][0]])
        if stop_tuple[0] == stop_tuple[1]:
            return
        else:
            return self.route_parallels(line, route, all_routes, bunching_value=self.bunching_value, line_spacing=self.line_spacing)

    def cluster_stops(self, stops_set, distance=100):
        """
        merges stops that are within distance together into one stop
        :param stops_set: iterable that lists stop_I's
        :param distance: int, distance to merge, meters
        :return:
        """
        df = self.gtfs.execute_custom_query_pandas("""SELECT * FROM stops
                                                 WHERE stop_I IN ({stops_set})""".format(stops_set=",".join([str(x) for x in stops_set])))
        df["geometry"] = df.apply(lambda row: Point((row["lon"], row["lat"])), axis=1)
        gdf = GeoDataFrame(df, crs=self.crs_wgs, geometry=df["geometry"])
        gdf = gdf.to_crs(self.crs_eurefin)
        gdf_poly = gdf.copy()
        gdf_poly["geometry"] = gdf_poly["geometry"].buffer(distance/2)
        gdf_poly["everything"] = 1

        gdf_poly = gdf_poly.dissolve(by="everything")

        polygons = None
        for geoms in gdf_poly["geometry"]:
            polygons = [polygon for polygon in geoms]

        single_parts = GeoDataFrame(crs=self.crs_eurefin, geometry=polygons)
        single_parts['new_stop_I'] = single_parts.index
        gdf_joined = sjoin(gdf, single_parts, how="left", op='within')
        single_parts["geometry"] = single_parts.centroid
        gdf_joined = gdf_joined.drop('geometry', 1)
        centroid_stops = single_parts.merge(gdf_joined, on="new_stop_I")
        return centroid_stops


        """
        change projection for accurate buffer distance
        merge polygons,
        select single parts
        calculate centroids
        """

    def plot_geopandas(self, gdf, **kwargs):
        fig, ax = plt.subplots()
        gdf.plot(column="rand", **kwargs)
        plt.show()

def main():
    """

    :return:
    """
    RMM = RouteMapMaker(gtfs_name)
    RMM.cluster_shapes()

if __name__ == "__main__":
    main()
