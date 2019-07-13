import os

import numpy as np
import pandas as pd
import matplotlib.mlab as ml
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
import networkx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geopandas import GeoDataFrame


from gtfspy.routing.journey_data import JourneyDataManager, DiffDataManager
from gtfspy.routing.journey_data_analyzer import JourneyDataAnalyzer
from research.westmetro_paper.scripts.all_to_all_settings import *
from research.westmetro_paper.scripts.all_to_all_analyzer import AllToAllDifferenceAnalyzer
from research.westmetro_paper.scripts.all_to_all_analyzer import stops_to_exclude

from gtfspy.mapviz_using_smopy_helper import *
from gtfspy.mapviz import plot_as_routes, plot_route_network_from_gtfs, plot_stops_with_attributes
from gtfspy.colormaps import *
from gtfspy.util import makedirs
from gtfspy.gtfs import GTFS
from gtfspy.route_types import *
from gtfspy.stats import get_section_stats
from gtfspy.networks import route_to_route_network

"""
Code for plotting o-d features on map.

"""
# TODO: Refactor this whole plotting thing:
"""
Keep the base (map) plotting functions general:
- plot points & plot lines
- support for ordinal & categorical variables
- input should contain lat/lon
- Parameters: color, size, output
- Cold be made as as class that handles an axis object combined with the smopy map -> 
all given coordinates are seamlessly converted to smopy coordinates, enabling easier handling of plotting several layers on the same map


Keep functions retrieving DB data general

MESO level runs the show--> What data to retrieve, data manipulation (adding coordinates, scaling values) --> what plotter to use

"""

TRAM = 0
SUBWAY = 1
RAIL = 2
BUS = 3
FERRY = 4

ROUTE_TYPE_TO_APPROXIMATE_CAPACITY = {
    TRAM: 74+14,
    SUBWAY: 65*4,
    RAIL: (232+28)*2,
    BUS: 47,
    FERRY: 200
}


def add_colorbar(im, aspect=10, pad_fraction=1, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    cb = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    return cb





def to_shapefile_point(df, name, lat_column="lat", lon_column="lon", dir="/home/clepe/production/results/helsinki/figs/shapefiles/"):
    df["geometry"] = df.apply(lambda row: Point((row[lon_column], row[lat_column])), axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df, crs=crs, geometry=df["geometry"])

    gdf.to_file(driver='ESRI Shapefile',
                filename=os.path.join(dir, name + ".shp"))


def plot_temporal_distances():
    target = TARGET_DICT["matinkylä"] # 7193 city
    origin = 7004 # 6929  # 3196 #3412 #3513 #7004 #3229 # 7913 #2865  # 1704  # 3523
    subplots = [221, 222]
    letters = {}
    fig = plt.figure()
    prev_ax = None
    for (gtfs_db, db_dict), subplot in zip(FEED_LIST, subplots):
        ax = fig.add_subplot(subplot, sharey=prev_ax)
        print(db_dict["gtfs_dir"], db_dict["journey_dir"])
        jdm = JourneyDataManager(db_dict["gtfs_dir"], journey_db_path=db_dict["journey_dir"],
                                 routing_params=routing_params(gtfs_db), track_vehicle_legs=TRACK_VEHICLE_LEGS,
                                 track_route=TRACK_ROUTE)
        npa = jdm.get_node_profile_time_analyzer(target, origin, db_dict["analysis_start_time"], db_dict["analysis_end_time"])
        ax, letters[db_dict["journey_dir"]] = npa.plot_temporal_distance_profile(jdm.gtfs.get_timezone_pytz(),
                                                                                 plot_journeys=True, ax=ax,
                                                                                 format_string="%H:%M:%S",
                                                                                 return_letters=True)
        ax.set_title(gtfs_db)
        prev_ax = ax
    prev_ax = None
    subplots = [223, 224]
    for (gtfs_db, db_dict), subplot in zip(FEED_LIST, subplots):
        gtfs = GTFS(db_dict["gtfs_dir"])
        letter_dict = letters[db_dict["journey_dir"]]
        ax = fig.add_subplot(subplot, sharey=prev_ax, sharex=prev_ax)
        jda = JourneyDataAnalyzer(db_dict["journey_dir"], db_dict["gtfs_dir"])
        df = jda.get_origin_target_journey_legs(origin, target,
                                                start_time=db_dict["analysis_start_time"],
                                                end_time=db_dict["analysis_end_time"],
                                                fastest_path=True)
        df = df.assign(color_attribute=lambda x: route_type_to_color_iterable(x.type))
        df = df.assign(zorder=lambda x: route_type_to_zorder(x.type))
        # TODO: currently only works for fastest path trips

        def unpack_times(all_times):
            labels = []
            for times in all_times:
                times = times.split(",")
                labels.append(",".join([letter_dict[int(time)] for time in times]))
            return labels

        df = df.assign(label=lambda x: unpack_times(x.dep_times))

        bounding_box = gtfs.get_bounding_box_by_stops([target, origin], 0.2)
        ax, smopy_map = plot_routes_as_stop_to_stop_network(numpy.array(df["from_lat"]),
                                                            numpy.array(df["from_lon"]),
                                                            numpy.array(df["to_lat"]),
                                                            numpy.array(df["to_lon"]),
                                                            attributes=numpy.absolute(
                                                                numpy.array([1 if x == 0 else x for x in df["n_trips"]])),
                                                            color_attributes=numpy.array(df["type"]),
                                                            zorders=numpy.array(df["zorder"]),
                                                            line_labels=list(df["label"]),
                                                            ax=ax,
                                                            return_smopy_map=True,
                                                            linewidth_multiplier=0.5,
                                                            use_log_scale=False,
                                                            alpha=1,
                                                            spatial_bounds=bounding_box,
                                                            c=None)
        prev_ax = ax

    plt.show()


def plot_travel_time_diff_per_section():
    """
    Calculate min, max, mean travel time per stop-to-stop section and calculate difference between dbs
    :return:
    """
    subplots = [211, 212]
    old_df = None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (gtfs_db, db_dict) in FEED_LIST:
        gtfs = GTFS(db_dict["gtfs_dir"])
        df = get_section_stats(gtfs)
        if old_df is None:
            old_df = df
    df = diff_df(old_df, df, ["from_stop_I", "to_stop_I", "type"], inner=True)
    df = gtfs.add_coordinates_to_df(df, stop_id_column="from_stop_I", lat_name="from_lat", lon_name="from_lon")
    df = gtfs.add_coordinates_to_df(df, stop_id_column="to_stop_I", lat_name="to_lat", lon_name="to_lon")
    df = df.assign(zorder=lambda x: route_type_to_zorder(x.type))
    color_values, norm, cmap = get_list_of_colors(df["min_time"], "delay_minutes")
    ax = plot_routes_as_stop_to_stop_network(df["from_lat"], df["from_lon"], df["to_lat"], df["to_lon"],
                                             color_attributes=color_values,
                                             zorders=df["zorder"], ax=ax)

    c = createcolorbar(cmap, norm)
    plt.show()


def plot_route_alternatives():
    for feed, feed_dict in FEED_LIST:
        jda = JourneyDataAnalyzer(feed_dict["journey_dir"], feed_dict["gtfs_dir"])
        gtfs = GTFS(feed_dict["gtfs_dir"])

        measures = [("simpson", 311), ("n_trips", 312), ("n_routes", 313)]
        for target in TARGET_STOPS:
            lat, lon = gtfs.get_stop_coordinates(target)
            print("creating figure for target: " + str(target))
            fig = plt.figure()
            plt.title("fastest path journey alternatives", fontsize=20)
            plt.axis('off')

            for (measure, subplot) in measures:
                ax = fig.add_subplot(subplot)
                df = jda.journey_alternatives_per_stop_pair(target,
                                                                feed_dict["analysis_start_time"],
                                                                feed_dict["analysis_end_time"])
                lats = df["lat"]
                lons = df["lon"]
                values = df[measure]
                cmap, norm = get_colormap(measure)
                ax, cax, smopy = plot_stops_with_attributes(lats, lons, values, alpha=0.5, colorbar=True,
                                                                  cmap=cmap, norm=norm, ax=ax)
                x, y = smopy.to_pixels(lat, lon)
                ax.scatter(x, y, s=20, c="g", marker="X", zorder=1)
                ax.set_title(measure)
                ax.axis('off')
                fig.colorbar(cax)

            zoom = 2
            w, h = fig.get_size_inches()
            fig.set_size_inches(w * zoom, h * zoom)
            img_dir = makedirs(os.path.join(FIGS_DIRECTORY))
            fig.savefig(os.path.join(img_dir, "journey_alternatives_" + feed +
                                     str(target)+".png"), format="png", dpi=300)


def plot_route_alternatives_diff():

    def get_merged_df():
        old_df = None
        old_feed = None
        df = None
        for feed, feed_dict in FEED_LIST:
            jda = JourneyDataAnalyzer(feed_dict["journey_dir"], feed_dict["gtfs_dir"])
            gtfs = GTFS(feed_dict["gtfs_dir"])
            df = jda.journey_alternative_data_time_weighted(target,
                                                            feed_dict["analysis_start_time"],
                                                            feed_dict["analysis_end_time"])
            df = df.set_index(["from_stop_I", "to_stop_I", "lat", "lon"])
            lat, lon = gtfs.get_stop_coordinates(target)
            if old_df is None:
                old_df = df
                old_feed = feed
            else:
                df = pd.merge(old_df, df, left_index=True, right_index=True, suffixes=[old_feed, feed])

                for column in ["simpson", "n_trips", "n_routes"]:
                    df["diff_"+column] = df[column+feed] - df[column+old_feed]

        df = df.reset_index()
        df = df[["from_stop_I", "to_stop_I", "lat", "lon", "diff_simpson", "diff_n_trips", "diff_n_routes"]]

        return lat, lon, df

    measures = [("diff_simpson", 311), ("diff_n_trips", 312), ("diff_n_routes", 313)]
    for target in TARGET_STOPS:
        print("creating figure for target: " + str(target))
        fig = plt.figure()
        plt.title("fastest path journey alternatives", fontsize=20)
        plt.axis('off')
        lat, lon, df = get_merged_df()
        for (measure, subplot) in measures:
            ax = fig.add_subplot(subplot)
            lats = df["lat"]
            lons = df["lon"]
            values = df[measure]
            cmap, norm = get_colormap(measure)
            ax, cax, smopy = plot_stops_with_attributes(lats, lons, values, alpha=0.5, colorbar=True,
                                                              cmap=cmap, norm=norm, ax=ax)
            x, y = smopy.to_pixels(lat, lon)
            ax.scatter(x, y, s=20, c="g", marker="X", zorder=1)
            ax.set_title(measure)
            ax.axis('off')
            fig.colorbar(cax)

        zoom = 2
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * zoom, h * zoom)
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY))
        fig.savefig(os.path.join(img_dir, "journey_alternatives_diff_" +
                                 str(target)+".png"), format="png", dpi=300)


def plot_stop_diff_maps():
    use_relative = False
    diff = DiffDataManager(DIFF_PATH)
    gtfs = GTFS(LM_DICT["gtfs_dir"])
    number_tuple = ("number", 1)
    minutes_tuple = ("minutes", 60)
    relative_tuple = ("relative", 1)
    multiples_tuple = ("multiples", 1)
    if use_relative:
        plot_parameters = {"n_boardings": multiples_tuple,
                           "journey_duration": relative_tuple,
                           "in_vehicle_duration": relative_tuple,
                           "transfer_wait_duration": relative_tuple,
                           "walking_duration": relative_tuple,
                           "temporal_distance": relative_tuple,
                           "pre_journey_wait_fp": relative_tuple}
        column_prefix = "rel_"
    else:
        plot_parameters = {"n_boardings": number_tuple,
                           "journey_duration": minutes_tuple,
                           "in_vehicle_duration": minutes_tuple,
                           "transfer_wait_duration": minutes_tuple,
                           "walking_duration": minutes_tuple,
                           "temporal_distance": minutes_tuple,
                           "pre_journey_wait_fp": minutes_tuple}
        column_prefix = ""

    measures = [("min", 221), ("max", 222), ("median", 223), ("mean", 224)]
    for target in TARGET_STOPS:
        lat, lon = gtfs.get_stop_coordinates(target)
        print("creating figure for target: " + str(target))
        for (plot_type, title_prefix) in [("diff_", ("relative " if use_relative else "")+"difference of ")]:
            for table, (unit, divisor) in plot_parameters.items():
                print("creating panels for: "+table)
                fig = plt.figure()
                plt.title(table, fontsize=20)
                plt.axis('off')

                for (measure, subplot) in measures:
                    ax = fig.add_subplot(subplot)
                    df = diff.get_table_with_coordinates(gtfs=gtfs, table_name=plot_type + table, target=target, 
                                                         use_relative=use_relative)
                    lats = df["lat"]
                    lons = df["lon"]
                    values = df[column_prefix+plot_type+measure].apply(lambda x: x/divisor)
                    cmap, norm = get_colormap(plot_type+unit)
                    ax, cax, smopy = plot_stops_with_attributes(lats, lons, values, alpha=0.5, colorbar=True,
                                                                      cmap=cmap, norm=norm, ax=ax)
                    x, y = smopy.to_pixels(lat, lon)
                    ax.scatter(x, y, s=10, c="g", marker="X", zorder=1)
                    ax.set_title(title_prefix+measure)
                    ax.axis('off')
                    fig.colorbar(cax)

                zoom = 2
                w, h = fig.get_size_inches()
                fig.set_size_inches(w * zoom, h * zoom)
                img_dir = makedirs(os.path.join(FIGS_DIRECTORY))
                fig.savefig(os.path.join(img_dir, table + ("_relative_" if use_relative else "_") +
                                         str(target)+".png"), format="png", dpi=300)

    if False:
        plt.scatter(df["lon"], df["lat"], c=df["diff_mean"], alpha=0.5)
        plt.colorbar()

        x = np.array(df["lon"])
        y = np.array(df["lat"])
        z = np.array(df["diff_mean"])
        xi = np.linspace(min(x), max(x))
        yi = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(xi, yi)

        Z = ml.griddata(x, y, z, xi, yi, interp='linear')
        plt.contour(X, Y, Z)
        plt.colorbar()

        plt.show()


def plot_largest_stop_diff_component_maps(targets=None):
    diff = DiffDataManager(DIFF_PATH)
    g = GTFS(LM_DICT["gtfs_dir"])
    plot_name = "component_with_largest_change_"
    title = "journey duration component with largest change"
    color_dict = {"pre_journey_wait": "r",
                  "in_vehicle_duration": "g",
                  "transfer_wait": "b",
                  "walking_duration": "c",
                  "no_change_within_threshold": "black"}
    if not targets:
        targets = TARGET_STOPS

    def colormap(keys, color_dict):

        return [color_dict[x] for x in keys]

    for target in targets:
        lat, lon = g.get_stop_coordinates(target)
        print("creating figure for target: " + str(target))

        fig = plt.figure()
        plt.title(title, fontsize=20)
        plt.axis('off')
        df = diff.get_largest_component(target)
        df = g.add_coordinates_to_df(df, stop_id_column="stop_I")
        lats = df["lat"]
        lons = df["lon"]
        subplots = [("increase", 211, "max_component"), ("decrease", 212, "min_component")]
        for (label, subplot, minmax) in subplots:
            ax = fig.add_subplot(subplot)
            ax.set_title(label)

            df = df.assign(color_attribute=lambda x: colormap(x[minmax], color_dict))
            ax, cax, smopy = plot_stops_with_attributes(lats, lons, df["color_attribute"], alpha=0.5, colorbar=True, ax=ax)
            x, y = smopy.to_pixels(lat, lon)
            ax.scatter(x, y, s=20, c="w", marker="X", zorder=1)
            ax.axis('off')

        plot_custom_label_point(color_dict)

        zoom = 2
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * zoom, h * zoom)
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY))
        fig.savefig(os.path.join(img_dir, plot_name +
                                 str(target)+".png"), format="png", dpi=300)

    if False:
        plt.scatter(df["lon"], df["lat"], c=df["diff_mean"], alpha=0.5)
        plt.colorbar()

        x = np.array(df["lon"])
        y = np.array(df["lat"])
        z = np.array(df["diff_mean"])
        xi = np.linspace(min(x), max(x))
        yi = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(xi, yi)

        Z = ml.griddata(x, y, z, xi, yi, interp='linear')
        plt.contour(X, Y, Z)
        plt.colorbar()

        plt.show()


def route_diff_2():
    gtfs = GTFS(OLD_DICT["gtfs_dir"])
    lm_gtfs = GTFS(LM_DICT["gtfs_dir"])
    old_df, lm_df = gtfs.get_route_difference_with_other_db(lm_gtfs, 7 * 3600, 9 * 3600, uniqueness_ratio=0.9)
    stops = gtfs.stops()

    def coords_from_string(string, stops_df, return_lats=True):
        route = string.split(',')

        lats = []
        lons = []
        for stop in route:
            row = stops_df.loc[stops_df['stop_I'] == int(stop)]
            lats.append(row.lat.item())
            lons.append(row.lon.item())
        if return_lats:
            return lats
        else:
            return lons

    old_df["lats"] = old_df.apply(lambda x: coords_from_string(x.route, stops, return_lats=True), axis=1)
    old_df["lons"] = old_df.apply(lambda x: coords_from_string(x.route, stops, return_lats=False), axis=1)
    lm_df["lats"] = lm_df.apply(lambda x: coords_from_string(x.route, stops, return_lats=True), axis=1)
    lm_df["lons"] = lm_df.apply(lambda x: coords_from_string(x.route, stops, return_lats=False), axis=1)

    fig = plt.figure()
    plt.title("routes before and after", fontsize=20)
    plt.axis('off')

    ax = fig.add_subplot(121)

    ax = plot_as_routes(old_df.to_dict(orient='records'), ax=ax, spatial_bounds=SPATIAL_BOUNDS, map_alpha=0.8, scalebar=True, legend=True,
                        return_smopy_map=False, line_width_attribute="n_trips", line_width_scale=0.1)
    ax = fig.add_subplot(122)

    ax = plot_as_routes(lm_df.to_dict(orient='records'), ax=ax, spatial_bounds=SPATIAL_BOUNDS, map_alpha=0.8, scalebar=True, legend=True,
                        return_smopy_map=False, line_width_attribute="n_trips", line_width_scale=0.1)
    plt.show()


def plot_route_diff_maps():
    """
    plots difference maps of two gtfs databases
    :return:
    """
    gtfs = GTFS(OLD_DICT["gtfs_dir"])
    lm_gtfs = GTFS(LM_DICT["gtfs_dir"])

    result = gtfs.get_section_difference_with_other_db(lm_gtfs, ANALYSIS_START_TIME_DS, ANALYSIS_END_TIME_DS)
    result = gtfs.add_coordinates_to_df(result, stop_id_column="from_stop_I", lat_name="from_lat", lon_name="from_lon")
    result = gtfs.add_coordinates_to_df(result, stop_id_column="to_stop_I", lat_name="to_lat", lon_name="to_lon")
    change = 0
    unch_or_incr = result.loc[result['diff_n_trips'] >= -1*change]
    unch_or_decr = result.loc[result['diff_n_trips'] <= change]

    increased = result.loc[result['diff_n_trips'] > change]
    decreased = result.loc[result['diff_n_trips'] < -1*change]
    for base_df, change_df, plot_name, c in zip([unch_or_decr, unch_or_incr],
                                                [increased, decreased],
                                                ["increase", "decrease"],
                                                ["blue", "red"]):
        fig = plt.figure()
        plt.title(plot_name+" in frequency", fontsize=20)
        plt.axis('off')
        print("plotting:", plot_name)
        ax = fig.add_subplot(111)
        ax, smopy_map = plot_routes_as_stop_to_stop_network(numpy.array(base_df["from_lat"]),
                                                            numpy.array(base_df["from_lon"]),
                                                            numpy.array(base_df["to_lat"]),
                                                            numpy.array(base_df["to_lon"]),
                                                            numpy.array(base_df["diff_n_trips"]), c="white", ax=ax,
                                                            return_smopy_map=True,
                                                            linewidth_multiplier=0.05,
                                                            spatial_bounds=SPATIAL_BOUNDS)
        for from_lat, from_lon, to_lat, to_lon, attribute in zip(numpy.array(change_df["from_lat"]),
                                                                 numpy.array(change_df["from_lon"]),
                                                                 numpy.array(change_df["to_lat"]),
                                                                 numpy.array(change_df["to_lon"]),
                                                                 numpy.array(change_df["diff_n_trips"])):
            xs, ys = smopy_map.to_pixels(numpy.array([from_lat, to_lat]), numpy.array([from_lon, to_lon]))
            ax.plot(xs, ys, c=c, linewidth=attribute*0.05)
        zoom = 2
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * zoom, h * zoom)
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY))
        fig.savefig(os.path.join(img_dir, plot_name + ".png"), format="png", dpi=300)


def plot_journey_routes_not_in_other_db():
    target = 7193
    use_log_scale = False
    all_leg_sections = False
    ignore_walk = True
    diff_threshold = None
    diff_path = None
    filter_by_travel_time_diff = True
    if filter_by_travel_time_diff:
        diff_threshold = 300
        diff_path = DIFF_PATH

    plot_name = "routes_not_in_other_db"
    if ignore_walk:
        plot_name += "no_walk_"
    for target in TARGET_STOPS:
        print(target)
        subplots = [211, 212]

        fig = plt.figure()
        prev_ax = None
        prev_df = None
        unique_types = []
        for (feed, feed_dict), subplot in zip(FEED_LIST, subplots):
            other_db_dict = FEED_DICT["lm_daily" if feed == "old_daily" else "old_daily"]
            other_jda = JourneyDataAnalyzer(other_db_dict["journey_dir"], other_db_dict["gtfs_dir"])
            gtfs = GTFS(LM_DICT["gtfs_dir"])  # THIS IS ONLY FOR STOP LIST
            ax = fig.add_subplot(subplot, sharey=prev_ax)
            jda = JourneyDataAnalyzer(feed_dict["journey_dir"], feed_dict["gtfs_dir"])
            df = jda.get_journey_routes_not_in_other_db(target,
                                                        other_jda.conn,
                                                        fastest_path=True,
                                                        ignore_walk=False)

            print("adding coordinates")
            df = gtfs.add_coordinates_to_df(df, stop_id_column="from_stop_I", lat_name="from_lat", lon_name="from_lon")

            df = gtfs.add_coordinates_to_df(df, stop_id_column="to_stop_I", lat_name="to_lat", lon_name="to_lon")
            df = df.assign(color_attribute=lambda x: route_type_to_color_iterable(x.type))

            df = df.assign(zorder=lambda x: route_type_to_zorder(x.type))
            print("preparing plot of", feed)
            ax, smopy_map = plot_routes_as_stop_to_stop_network(numpy.array(df["from_lat"]),
                                                                numpy.array(df["from_lon"]),
                                                                numpy.array(df["to_lat"]),
                                                                numpy.array(df["to_lon"]),
                                                                attributes=numpy.absolute(numpy.array(
                                                                   [1 if x == 0 else x for x in df["n_trips"]])),
                                                                color_attributes=numpy.array(df["color_attribute"]),
                                                                zorders=numpy.array(df["zorder"]),
                                                                ax=ax,
                                                                return_smopy_map=True,
                                                                linewidth_multiplier=0.001,
                                                                use_log_scale=use_log_scale,
                                                                alpha=0.2,
                                                                spatial_bounds=SPATIAL_BOUNDS)
            lat, lon = gtfs.get_stop_coordinates(target)
            x, y = smopy_map.to_pixels(lat, lon)
            ax.scatter(x, y, s=30, c="white", marker="X", zorder=20)
            if prev_df is not None:
                new_df = df["type"].append(prev_df["type"])
                unique_types = list(new_df.unique())

            ax.set_title(feed)
            prev_df = df
            prev_ax = ax

        plot_custom_label_line(unique_types, use_log_scale)

        zoom = 2
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 2 * zoom, h * zoom)
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY))
        print("saving image")

        fig.savefig(os.path.join(img_dir, plot_name + str(target) + ".png"),
                    format="png", dpi=300)


def plot_journeys_to_target():
    targets = [1040]
    use_log_scale = True
    all_leg_sections = False
    ignore_walk = True
    diff_threshold = None
    diff_path = None
    filter_by_travel_time_diff = False
    if filter_by_travel_time_diff:
        diff_threshold = 300
        diff_path = DIFF_PATH

    plot_name = "all_journey_sections_to_target_" if all_leg_sections else "journey_legs_to_target_"
    if ignore_walk:
        plot_name += "no_walk_"
    for target in targets:
        print(target)
        subplots = [131, 132, 133]

        fig = plt.figure()
        prev_ax = None
        prev_df = None
        unique_types = []
        dfs = {}
        for gtfs_db, feed_dict in FEED_LIST:
            print("retrieving data for", gtfs_db)
            jda = JourneyDataAnalyzer(feed_dict["journey_dir"], feed_dict["gtfs_dir"])
            dfs[gtfs_db] = jda.get_journey_legs_to_target(target=target,
                                                          fastest_path=True,
                                                          min_boardings=False,
                                                          all_leg_sections=all_leg_sections,
                                                          ignore_walk=ignore_walk,
                                                          diff_threshold=diff_threshold,
                                                          diff_path=diff_path)
        print("merging results")
        dfs["diff"] = diff_df(dfs["old_daily"], dfs["lm_daily"], index_columns=["from_stop_I", "to_stop_I", "type"])

        for i, subplot in zip(["old_daily", "diff", "lm_daily"], subplots):
            gtfs = GTFS(LM_DICT["gtfs_dir"])
            ax = fig.add_subplot(subplot, sharey=prev_ax)
            df = dfs[i]
            if i == "diff":
                df = df.assign(color_attribute=lambda x: ['b' if x < 0 else 'r' if x > 0 else 'w' for x in x.n_trips])
            else:
                df = df.assign(color_attribute=lambda x: route_type_to_color_iterable(x.type))

            print("adding coordinates")
            df = gtfs.add_coordinates_to_df(df, stop_id_column="from_stop_I", lat_name="from_lat", lon_name="from_lon")

            df = gtfs.add_coordinates_to_df(df, stop_id_column="to_stop_I", lat_name="to_lat", lon_name="to_lon")

            df = df.assign(zorder=lambda x: route_type_to_zorder(x.type))
            print("preparing plot of", i)
            ax, smopy_map = plot_routes_as_stop_to_stop_network(numpy.array(df["from_lat"]),
                                                                numpy.array(df["from_lon"]),
                                                                numpy.array(df["to_lat"]),
                                                                numpy.array(df["to_lon"]),
                                                                attributes=numpy.absolute(numpy.array([1 if x == 0 else x for x in df["n_trips"]])),
                                                                color_attributes=numpy.array(df["type"]),
                                                                zorders=numpy.array(df["zorder"]),
                                                                ax=ax,
                                                                return_smopy_map=True,
                                                                linewidth_multiplier=1,
                                                                use_log_scale=use_log_scale,
                                                                alpha=0.2,
                                                                spatial_bounds=SPATIAL_BOUNDS)
            lat, lon = gtfs.get_stop_coordinates(target)
            x, y = smopy_map.to_pixels(lat, lon)
            ax.scatter(x, y, s=30, c="white", marker="X", zorder=20)
            if prev_df is not None:
                new_df = df["type"].append(prev_df["type"])
                unique_types = list(new_df.unique())

            ax.set_title(i)
            prev_df = df
            prev_ax = ax

        plot_custom_label_line(unique_types, use_log_scale)

        zoom = 2
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 2 * zoom, h * zoom)
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY))
        print("saving image")

        fig.savefig(os.path.join(img_dir, ("thresholded_" if filter_by_travel_time_diff else "") + plot_name + str(target) + ".png"), format="png", dpi=300)


def plot_custom_label_line(unique_types, use_log_scale):
    handles = []
    labels = []
    linewidths = [1, 10, 100, 1000]
    for linewidth in linewidths:
        if use_log_scale:
            orig_linewidth = linewidth
            linewidth = math.log10(linewidth)
            line = mlines.Line2D([], [], color="Black", linewidth=linewidth,
                                 label=str(orig_linewidth))
            handles.append(line)
            labels.append(str(orig_linewidth))
    for type in unique_types:
        line = mlines.Line2D([], [], color=ROUTE_TYPE_TO_COLOR[type], linewidth=10,
                             label=ROUTE_TYPE_TO_SHORT_DESCRIPTION[type])
        handles.append(line)
        labels.append(ROUTE_TYPE_TO_SHORT_DESCRIPTION[type])

    plt.figlegend(labels=labels, handles=handles, loc='upper right', ncol=1)


def plot_custom_label_point(types_color_dict):
    handles = []
    labels = []

    for type, color in types_color_dict.items():
        point = mpatches.Patch(color=color, label=type)
        handles.append(point)
        labels.append(type)

    plt.figlegend(labels=labels, handles=handles, loc='upper right', ncol=1)


def plot_route_maps():
    for (feed, feed_dict) in FEED_LIST:
        g = GTFS(feed_dict["gtfs_dir"])

        plot_route_network_from_gtfs(g)
        plt.show()
        plt.savefig(os.path.join(FIGS_DIRECTORY, feed + ".png"), format="png", dpi=300)


def plot_weighted_route_maps():
    """
    Modular approach based on stop segments
    - find stop clusters,
    - find trip counts separately for each type
    - find representative shape for segment
    - width by, color by
    :return:
    """
    use_seats = False
    for (feed, feed_dict) in FEED_LIST:
        gtfs = GTFS(feed_dict["gtfs_dir"])

        query = (
            " SELECT q1.stop_I as from_stop_I, q2.stop_I as to_stop_I, q1.trip_I as trip_I, COUNT(*) as freq FROM"
            " (SELECT * FROM stop_times) q1,"
            " (SELECT * FROM stop_times) q2"
            " WHERE q1.seq+1=q2.seq AND q1.trip_I=q2.trip_I"
            " GROUP BY from_stop_I, to_stop_I"
            " ORDER BY COUNT(*) DESC")
        df = gtfs.execute_custom_query_pandas(query)
        query_type = (
            " SELECT  t.trip_I as trip_I, t.route_I, r.type as type FROM trips AS t"
            " JOIN"
            " (SELECT * FROM routes) as r"
            " ON r.route_I = t.route_I")

        df_type = gtfs.execute_custom_query_pandas(query_type)

        df = pd.merge(df, df_type, on="trip_I", how="inner")

        from_lats = []
        from_lons = []
        for i in range(len(df.from_stop_I)):
            lat, lon = gtfs.get_stop_coordinates(stop_I=df.from_stop_I[i])
            from_lats.append(lat)
            from_lons.append(lon)

        to_lats = []
        to_lons = []
        for i in range(len(df.to_stop_I)):
            lat, lon = gtfs.get_stop_coordinates(stop_I=df.to_stop_I[i])
            to_lats.append(lat)
            to_lons.append(lon)
        attributes = df.freq.tolist()

        if use_seats:
            types = list(df.type)
            attributes = [x*ROUTE_TYPE_TO_APPROXIMATE_CAPACITY[t] for x, t in zip(attributes, types)]

        plt.figure(figsize=(5, 6))
        ax = plt.subplot(111, projection="smopy_axes")
        df = df.assign(color_attribute=lambda x: x.type)

        df = df.assign(zorder=lambda x: route_type_to_zorder(x.type))
        ax = plot_routes_as_stop_to_stop_network(from_lats, from_lons, to_lats, to_lons, attributes=attributes,
                                                 use_log_scale=False, color_attributes=df.color_attribute,
                                                 linewidth_multiplier=0.00005 if use_seats else 0.005, alpha=0.5, ax=ax,
                                                 spatial_bounds=SPATIAL_BOUNDS,
                                                 legend_multiplier=100 if use_seats else 1,
                                                 legend_unit="seats/day" if use_seats else "veh./day")
        #plt.show()
        plt.savefig(os.path.join(FIGS_DIRECTORY, feed + ("_seats" if use_seats else "") + "_map.pdf"), format="pdf", dpi=300, bbox_inches='tight')


def diff_df(old_df, new_df, index_columns, column_prefix="", return_all=False, inner=False):
    """
    Merges two dataframes with the same columns using the index columns while subtracting the other column values
    :param old_df: Pandas Dataframe
    :param new_df: Pandas Dataframe
    :param index_columns: list of columns
    :param column_prefix: string, prefix to add to the new columns
    :param return_all: Bool, if to return the original columns or only the subtracted
    :return:
    """
    old_suffix = "_old"
    new_suffix = "_new"
    old_df = old_df.set_index(index_columns)
    new_df = new_df.set_index(index_columns)
    common_columns = [x for x in list(old_df) if x in list(new_df)]
    if inner:
        how = 'inner'
    else:
        how = 'outer'
    df = pd.merge(old_df, new_df, left_index=True, right_index=True, suffixes=[old_suffix, new_suffix], how=how)
    df = df.fillna(0)
    for column in common_columns:
        df[column_prefix + column] = df[column + new_suffix] - df[column + old_suffix]

    if not return_all:
        columns = [x for x in list(df) if not old_suffix in x and not new_suffix in x]
        df = df[columns]

    df = df.reset_index()
    return df


def plot_origin_destination_lines():
    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, A2AA_DB_OLD_PATH, A2AA_DB_LM_PATH, A2AA_OUTPUT_DB_PATH)
    gtfs = GTFS(GTFS_PATH)
    df = a2aa.extreme_change_od_pairs(600)
    df = gtfs.add_coordinates_to_df(df, stop_id_column="from_stop_I", lat_name="from_lat", lon_name="from_lon")
    df = gtfs.add_coordinates_to_df(df, stop_id_column="to_stop_I", lat_name="to_lat", lon_name="to_lon")
    print("preparing to plot")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plot_routes_as_stop_to_stop_network(df["from_lat"], df["from_lon"], df["to_lat"], df["to_lon"],
                                             color_attributes=None,
                                             c="b", alpha=0.1, ax=ax)
    plt.show()


def a2a_change_map(time=7, groupby="to_stop_I", measure="temporal_distance", column="mean_diff_mean", s=6, img_dir=None, ignore_stops=False):
    """
    Calculates the mean change of temporal distance or number of boardings in a all-to-all routing
    :return:
    """

    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                      get_a2aa_db_path(time, "output"))
    ignore_list=None
    if ignore_stops:
        ignore_list = stops_to_exclude(return_sqlite_list=True)
    df = a2aa.get_mean_change_for_all_targets(groupby=groupby, measure=measure, ignore_stops=ignore_list)
    fig = plt.figure()

    #plt.title("", fontsize=20)
    #plt.axis('off')
    ax = fig.add_subplot(111)
    lats = df["lat"]
    lons = df["lon"]
    values = df[column]
    #to_shapefile_point(df, groupby+measure+column)
    if "cr_count_" in column:
        cmap, norm = get_colormap("n_locations")
    elif measure == "temporal_distance":
        #cmap, norm = get_colormap("diff_minutes")
        cmap, norm = get_colormap_with_params(-15, 15, name="RdBu_r")
    else:
        cmap, norm = get_colormap_with_params(-1, 1, name="RdBu_r")
    ax, cax, smopy = plot_stops_with_attributes(lats, lons, values, s=s, alpha=1, colorbar=True,
                                                      cmap=cmap, norm=norm, ax=ax, scalebar=True)
    #ax.axis('off')
    cb = add_colorbar(cax)
    cb.ax.tick_params(labelsize=15)

    #ax2 = fig.add_axes([.9, 0.1, 0.03, 0.8]) #_subplot(122) #
    #cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional')
    plt.xticks([], [])
    plt.yticks([], [])

    if measure == "temporal_distance":
        unit = "minutes"
    else:
        unit = "boardings"
    #ax.yaxis.set_label_position("right")
    #ax.yaxis.labelpad = 70
    cb.set_label(unit, size=20)
    if False:
        xs = []
        ys = []
        for lat, lon in zip(np.array(df["lat"]), np.array(df["lon"])):
            x, y = smopy.to_pixels(lat, lon)
            xs.append(x)
            ys.append(y)
        z = np.array(df["mean_diff_mean"])
        xi = np.linspace(min(xs), max(xs))
        yi = np.linspace(min(ys), max(ys))
        X, Y = np.meshgrid(xi, yi)

        Z = ml.griddata(xs, ys, z, xi, yi, interp='linear')
        CS = plt.contour(X, Y, Z, linewidths=1)
        plt.clabel(CS, inline=1, fontsize=10)

        plt.colorbar()

    # plt.show()

    zoom = 2
    w, h = fig.get_size_inches()

    fig.tight_layout()
    fig.set_size_inches(w * zoom, h * zoom)
    if not img_dir:
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY, "a2a_change_maps"))
    fig.savefig(os.path.join(img_dir, "mean_change_" + groupby + "_" + measure + "_" + column + "_time_" + str(time) + ".pdf"), format="pdf", dpi=300, bbox_inches='tight')


def a2a_change_map_smopy(time=7, groupby="to_stop_I", measure="temporal_distance", column="mean_diff_mean", s=6, img_dir=None, ignore_stops=False):
    """
    Calculates the mean change of temporal distance or number of boardings in a all-to-all routing
    :return:
    """

    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                      get_a2aa_db_path(time, "output"))
    ignore_list=None
    if ignore_stops:
        ignore_list = stops_to_exclude(return_sqlite_list=True)
    df = a2aa.get_mean_change_for_all_targets(groupby=groupby, measure=measure, ignore_stops=ignore_list)
    fig = plt.figure(figsize=[4.4, 4])
    ax = fig.add_subplot(111, projection="smopy_axes")

    lats = df["lat"]
    lons = df["lon"]
    values = df[column]
    #to_shapefile_point(df, groupby+measure+column)
    if "cr_count_" in column:
        cmap, norm = get_colormap("n_locations")
    elif measure == "temporal_distance":
        #cmap, norm = get_colormap("diff_minutes")
        cmap, norm = get_colormap_with_params(-15, 15, name="RdBu_r")
    else:
        cmap, norm = get_colormap_with_params(-1, 1, name="RdBu_r")
    ax, cax = plot_stops_with_attributes_smopy(lats, lons, values, ax=ax, s=s, alpha=0.5,
                                               cmap=cmap, norm=norm, scalebar=True)
    ax.set_plot_bounds(**SPATIAL_BOUNDS)
    ax.autoscale(False)
    cb = add_colorbar2(cax, ax)

    cb.ax.tick_params(labelsize=7)

    if measure == "temporal_distance":
        unit = "minutes"
    else:
        unit = "transfers"

    cb.set_label(unit, size=10)
    if False:
        xs = []
        ys = []
        for lat, lon in zip(np.array(df["lat"]), np.array(df["lon"])):
            x, y = smopy.to_pixels(lat, lon)
            xs.append(x)
            ys.append(y)
        z = np.array(df["mean_diff_mean"])
        xi = np.linspace(min(xs), max(xs))
        yi = np.linspace(min(ys), max(ys))
        X, Y = np.meshgrid(xi, yi)

        Z = ml.griddata(xs, ys, z, xi, yi, interp='linear')
        CS = plt.contour(X, Y, Z, linewidths=1)
        plt.clabel(CS, inline=1, fontsize=10)

        plt.colorbar()
    if not img_dir:
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY, "a2a_change_maps"))
    fig.savefig(os.path.join(img_dir, "mean_change_" + groupby + "_" + measure + "_" + column + "_time_" + str(time) + ".pdf"), format="pdf", dpi=300, bbox_inches='tight')


def plot_route_to_route_network():
    """Plots a graph where nodes are bus routes, edges indicate possible transfer"""
    gtfs = GTFS(GTFS_PATH)
    pos_df = gtfs.execute_custom_query_pandas("""SELECT route_id, (max(lat) + min(lat))/2 AS lat, (max(lon) + min(lon))/2 AS lon 
                                          FROM routes, trips, stop_times, stops
                                          WHERE routes.route_I=trips.route_I AND trips.trip_I=stop_times.trip_I AND stop_times.stop_I=stops.stop_I
                                          GROUP BY routes.route_I""")
    pos = {i.route_id: (i.lon, i.lat) for i in pos_df.itertuples()}
    graph = route_to_route_network(gtfs, 200, 7*3600, 8*3600)
    nodelist = graph.nodes()
    node_colors = networkx.get_node_attributes(graph, 'color')
    color_list = [node_colors[i] for i in nodelist]
    networkx.draw_networkx(graph, pos=pos, node_size=500, nodelist=nodelist, node_color=color_list)
    plt.draw()
    plt.show()


def plot_df_point_data(values, lats, lons, filename, spatial_bounds=None, img_dir=None,
                       color_map_params=None,
                       annotates=None,
                       location_markers=None):
    """
    Plots one thing in set path
    :return:
    """
    if annotates and not isinstance(annotates, list):
        annotates = [annotates]
    if location_markers and not isinstance(location_markers, list):
        location_markers = [location_markers]
    fig = plt.figure(frameon=False)

    plt.title("", fontsize=20)
    plt.axis('off')
    ax = fig.add_subplot(111)
    ax = fig.add_axes([0, 0, 1, 1])
    cmap, norm = get_colormap_with_params(color_map_params["min"], color_map_params["max"], name=color_map_params["name"])
    ax, cax, smopy_map = plot_stops_with_attributes_smopy(lats, lons, values, s=10, alpha=1, colorbar=True, spatial_bounds=spatial_bounds,
                                                          cmap=cmap, norm=norm, ax=ax)
    ax.axis('off')

    if False:
        xs = []
        ys = []
        for lat, lon in zip(np.array(lats), np.array(lons)):
            x, y = smopy_map.to_pixels(lat, lon)
            xs.append(x)
            ys.append(y)
        z = np.array(values)
        xi = np.linspace(min(xs), max(xs))
        yi = np.linspace(min(ys), max(ys))
        X, Y = np.meshgrid(xi, yi)

        Z = ml.griddata(xs, ys, z, xi, yi, interp='linear')
        CS = plt.contour(X, Y, Z, linewidths=1)
        plt.clabel(CS, inline=1, fontsize=10)

        plt.colorbar()

    # plt.show()
    if annotates:
        for annotate in annotates:
            xs, ys = smopy_map.to_pixels(annotate["lats"], annotate["lons"])
            #ax.scatter(xs, ys, c="white", marker="x", s=20)
            for text, x, y in zip(annotate["label"], xs, ys):
                ax.text(x, y, text, size=20, color=annotate["color"],
                        bbox={'facecolor': 'black', 'alpha': 0.3, 'pad': 0})
    if location_markers:
        for location_marker in location_markers:
            xs, ys = smopy_map.to_pixels(location_marker["lats"], location_marker["lons"])
            ax.scatter(xs, ys, c="cyan", marker="x", s=600)

    fig.tight_layout()
    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    if not img_dir:
        img_dir = makedirs(os.path.join(FIGS_DIRECTORY, "a2a_change_maps"))
    fig.savefig(os.path.join(img_dir, filename + ".pdf"), format="pdf", dpi=300)
    fig, ax = plt.subplots()
    plt.colorbar(cax,  orientation="horizontal", fraction=0.03, pad=0.01)
    ax.remove()
    plt.savefig(os.path.join(img_dir, ("n_boardings" if "n_boardings" in filename else "temporal_distance") + "_colorbar" + ".png"), format="png",bbox_inches='tight')


def plot_markers_on_map(lats, lons, ax, smopy_map, markers=None, labels=None, c=None, s=None, alpha=None):

    xs, ys = smopy_map.to_pixels(lats, lons)

    cax = ax.scatter(xs, ys, c=c, s=s, marker=markers, alpha=alpha)
    ax.annotate(labels, (lats, lons))


"""
mean_change_map(groupby="to_stop_I", measure="temporal_distance")
mean_change_map(groupby="from_stop_I", measure="temporal_distance")
mean_change_map(groupby="to_stop_I", measure="n_boardings")
mean_change_map(groupby="from_stop_I", measure="n_boardings")
"""
if __name__ == "__main__":
    plot_weighted_route_maps()