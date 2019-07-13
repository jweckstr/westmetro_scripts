"""
Measures to calculate:
    Table of transit supply change
    Aggregated travel time and boardings
    Map of mean travel time/boardings change - exists, however should the boardings be made using min boardings trips
    Heatmap - Mean travel time vs. change zonewise
    Distribution - change zonewise
    Relative difference winner stops
    Relative difference losers stops
    Travel impedance components?

Optional TODO: generalized cost mechanic: how is the actual travel time changing when considering different values for boardings
"""

import pandas as pd

from gtfspy.mapviz_using_smopy_helper import add_colorbar2

from research.westmetro_paper.scripts.map_plots import *
from research.westmetro_paper.scripts.basestats import *
from research.westmetro_paper.scripts.zone_analysis import *
from gtfspy.gtfs import GTFS

output_path = "/home/clepe/git/7589451nktxprytvbdx/figs"

# Table of transit supply change -  See basestats

# Aggregated travel time and boardings

# routemaps: plot_weighted_route_maps in map_plots.py

def seconds_to_hour_minutes_string(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# TABLE
if False:
    measures = ["temporal_distance", "n_boardings"]
    master_dfs = {}
    merged_df = None
    for measure in measures:
        master_df = pd.DataFrame()

        for time in TIMES:

            a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                              get_a2aa_db_path(time, "output"))
            df = a2aa.get_global_mean_change(measure, ignore_stops=True)
            df["time"] = str(time)+" - "+str(time+1)
            master_df = master_df.append(df)
        #master_df["percent_change"] = master_df.apply(lambda row: row.global_mean_difference/row.before_global_mean, axis=1)
        for column in master_df.columns:
            if not column == 'time' and not column == 'percent_change' and not measure == 'n_boardings':
                master_df[column] = master_df[column].apply(lambda x: seconds_to_hour_minutes_string(x))
        master_df = master_df.round(3)
        #master_df = master_df.set_index(["time"])
        if measure == 'n_boardings':
            master_df.columns = ['Mean MNBA, before', 'Mean MNBA, after', 'Mean MNBA, change', 'time']
        if measure == 'temporal_distance':
            master_df.columns = ['Mean MTTA, before',  'Mean MTTA, after', 'Mean MTTA, change', 'time']
        print(master_df)
        master_dfs[measure] = master_df
    for measure, df in master_dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on="time")
    merged_df = merged_df[["time"]+[x for x in merged_df.columns if not "time" == x]]
    print(merged_df.to_latex(index=False))

# Map of mean travel time/boardings change
if False:
    """min_diff_mean, mean_diff_mean, max_diff_mean"""
    for time in TIMES:
        for groupby in ["from_stop_I"]:
            for measure in ["n_boardings", "temporal_distance"]:
                a2a_change_map_smopy(time=time, groupby=groupby, measure=measure, column="mean_diff_mean", img_dir=None, ignore_stops=True)

# Heatmap - Mean travel time vs. change zonewise - See zone_analysis

if True:
    from matplotlib import rcParams
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    mode = "temporal_distance"  # "n_boardings" # "temporal_distance"
    for time in TIMES[:1]:
        rerun = False
        a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                          get_a2aa_db_path(time, "output"))
        #heatmap_singles(a2aa, measure="mean", rerun=True, img_dir=None)
        histogram_matrix(a2aa, measure="mean", mode=mode, rerun=rerun, img_dir=None, fig_name="hist_"+mode+str(time))
        ba_histogram_matrix(a2aa, measure="mean", measure_mode=mode, rerun=rerun, img_dir=None, fig_name="ba_"+mode+str(time))

    plt.show()


# Distribution - change zonewise - See analysis_zones

# Relative difference winner stops
# Relative difference losers stops


if False:
    #zone_scatter(img_dir=output_path)
    for time in TIMES:
        a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                          get_a2aa_db_path(time, "output"))
        old_vs_change_scatter_colored_by_zone(a2aa, file_id=time)
# Winner/Loser stop selection?
# Distribution of change
# cut the flesh: find a method to ignore the stops in the perifery
# Remove stops which have a high proportion of unreachable (inf) stops?
# Limit to Helsinki and Espoo and the feeder bus zone + new metro stations, and define walking zone with same analogy as feeder bus zone
# cap at 2 hours as that is the effective routing time

if False:
    # jana
    time = 7
    targets = []
    direction = "to"
    stops = TARGET_DICT
    img_dir = makedirs(os.path.join(FIGS_DIRECTORY, "a2a_change_maps", "jana1"))
    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                      get_a2aa_db_path(time, "output"))
    gtfs = GTFS(GTFS_PATH)
    measures = ["n_boardings", "temporal_distance"]
    indicators = ["diff_mean"]

    for measure in measures:
        for indicator in indicators:
            for location, stop in stops.items():
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="smopy_axes")
                if measure == "n_boardings":
                    yedges = np.arange(-2.0, 2.0, 0.1)
                    unit = "s"
                else:
                    yedges = range(-25, 25, 1)
                    unit = "m"
                if indicator == "diff_mean_relative":
                    yedges = np.arange(-0.7, 0.7, 0.05)
                    unit = "s"
                df = a2aa.get_data_for_target(stop, measure, direction=direction, unit=unit, ignore_stops=True)
                df = gtfs.add_coordinates_to_df(df, stop_id_column=("to" if direction == 'from' else 'from') + '_stop_I', lat_name="lat", lon_name="lon")
                color_map_params = {"name": "RdBu_r", "min": -2, "max": 2} if measure == "n_boardings" else {"name": "RdBu_r", "min": -30, "max": 30}
                cmap, norm = get_colormap_with_params(color_map_params["min"], color_map_params["max"],
                                                      name=color_map_params["name"])
                stuff = ax.scatter(df['lon'], df['lat'], c=df[indicator], s=10, cmap=cmap, norm=norm, zorder=1)
                jana = "DEF"
                if TARGET_LETTERS[location] in "ABC":
                    jana = "ABC"
                jana_stops = {loc: stop for loc, stop in stops.items() if TARGET_LETTERS[loc] in jana}
                lats = [gtfs.get_stop_coordinates(stop)[0] for loc, stop in jana_stops.items()]
                lons = [gtfs.get_stop_coordinates(stop)[1] for loc, stop in jana_stops.items()]
                ax.plot(lons, lats, linewidth=2, c="black", zorder=2)

                for location2, stop2 in jana_stops.items():
                    lat, lon = gtfs.get_stop_coordinates(stop2)
                    ax.scatter(lon, lat, s=50, c="black", zorder=3)
                    if not stop2 == stop:
                        ax.scatter(lon, lat, s=30, c="white", zorder=4)
                    if stop2 == stop:
                        ax.scatter(lon, lat, s=50, c="black", zorder=5)

                ax.set_plot_bounds(**SPATIAL_BOUNDS)
                # scalebar and colorbar
                kws = {"frameon": False, "location": "lower right"}
                ax.add_scalebar(**kws)
                fig.savefig(os.path.join(img_dir,
                                         measure + indicator + "_" + location + ".pdf"), format="pdf", dpi=300, bbox_inches='tight')
                fig, ax = plt.subplots()
                cb = add_colorbar2(stuff, ax=ax, drop_ax=True)
                #plt.colorbar(stuff, ax=ax)
                cb.ax.tick_params(labelsize=20)
                cb.set_label("transfers" if measure == "n_boardings" else "minutes", size=20)
                plt.savefig(os.path.join(img_dir, measure +"_colorbar.pdf"), format="pdf", dpi=300, bbox_inches='tight')

                plt.clf()

if False:
    # map for each stop
    time = 7
    direction = "to"
    stops = TARGET_DICT
    img_dir = makedirs(os.path.join(FIGS_DIRECTORY, "a2a_change_maps"))
    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                      get_a2aa_db_path(time, "output"))
    gtfs = GTFS(GTFS_PATH)
    measures = ["n_boardings", "temporal_distance"]
    indicators = ["diff_mean_relative", "diff_mean"]
    row_list = []



    #separate CDF plot
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(stops.values())))

    for measure in measures:
        for indicator in indicators:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, title="")
            for (location, stop), color in zip(stops.items(), colors):
                row_dict = {"Location": location,
                            "Location_id": stop,
                            "MTTA, map": "temporal_distance_to_stop_" + str(stop) + ".pdf",
                            "MTTA, distribution": stop,
                            "MNBA, map": "n_boardings_to_stop_" + str(stop) + ".pdf",
                            "MNBA, distribution": stop}
                row_list.append(row_dict)

                lat, lon = gtfs.get_stop_coordinates(stop)

                ax = single_stop_change_histogram(stop, measure, ax=ax, indicator=indicator, direction=direction, a2aa=a2aa,
                                                    img_dir=img_dir, return_ax=True, cdf=True, color=color, label=location)

                if measure == "n_boardings":
                    yedges = np.arange(-2.0, 2.0, 0.1)
                    unit = "s"
                else:
                    yedges = range(-25, 25, 1)
                    unit = "m"
                if indicator == "diff_mean_relative":
                    yedges = np.arange(-0.7, 0.7, 0.05)
                    unit = "s"
                df = a2aa.get_data_for_target(stop, measure, direction=direction, unit=unit, ignore_stops=True)
                df = gtfs.add_coordinates_to_df(df, stop_id_column=("to" if direction == 'from' else 'from') + '_stop_I', lat_name="lat", lon_name="lon")

                plot_df_point_data(df["diff_mean"], df['lat'], df['lon'], measure + '_' + direction+'_stop_' + str(stop), img_dir=img_dir, color_map_params={"name": "RdBu", "min": -2, "max": 2} if measure == "n_boardings" else
                               {"name": "seismic", "min": -30, "max": 30}, location_markers={'stop': stop, 'lats': lat, 'lons': lon})
            ax.legend()
            ax.set_xlabel("NMBA" if measure == "n_boardings" else "MTTA")
            ax.set_ylabel("stops")
            fig.savefig(os.path.join(img_dir,
                                     measure+indicator+"cdfs.png"), format="png", dpi=300)

            df = pd.DataFrame(row_list)
            df_as_latex = df[["Location",
                              "Location_id",
                              "MTTA, map",
                              #"MTTA, distribution",
                              "MNBA, map",
                              #"MNBA, distribution"
                               ]].to_latex(index=False)
            print(df_as_latex)


if False:
    time = 7
    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                      get_a2aa_db_path(time, "output"))
    gtfs = GTFS(GTFS_PATH)
    top_n = 5
    img_dir = "/home/clepe/production/results/helsinki/figs/all_to_all"
    measures = ["n_boardings", "temporal_distance"]
    indicators = ["diff_mean_relative", "diff_mean"]
    all_dfs, _ = analysis_zones(as_dict=True)
    include_list = pd.concat([all_dfs["new_metro_stations"], all_dfs["feeder_bus_area"]])
    for measure in measures:
        for indicator in indicators:
            annotates = []

            fig, axes2d = plt.subplots(nrows=top_n, ncols=2, sharex=True, sharey=True, figsize=(10, 8))
            new_order = []
            for i in range(len(axes2d[0])):
                column = []
                for row in axes2d:
                    column.append(row[i])
                new_order.append(column)
            axes2d = new_order

            for losers, row in zip([True, False], axes2d):

                df = a2aa.get_n_winning_targets_using_change_in_mean(top_n, measure, losers=losers, threshold=10800,
                                                                     include_list=include_list["stop_I"])

                df = gtfs.add_coordinates_to_df(df, stop_id_column="to_stop_I", lat_name="lat", lon_name="lon")
                # plot_df_point_data()
                for item, cell in zip(df.itertuples(), row):
                    cell = single_stop_change_histogram(item.to_stop_I, measure, indicator=indicator, direction="to", a2aa=a2aa,
                                                        img_dir=output_path, ax=cell, return_ax=True)
                annotates.append({
                    "lats": df["lat"],
                    "lons": df["lon"],
                    "label": df["to_stop_I"].apply(lambda x: ("L " if losers else "W ") + str(x)),
                    "color": "white"
                })

                df_to_print = df.round(2)
                if measure == "temporal_distance":
                    df_to_print["diff_mean"] = df_to_print["diff_mean"].apply(lambda x: x/60)
                df_to_print["filename"] = df_to_print["to_stop_I"].apply(lambda x: str(x))
                df_to_print = df_to_print.sort_values("diff_mean", ascending=not losers)

                df_as_latex = df_to_print[["to_stop_I", "diff_mean", "filename"]].to_latex(index=False)
                print(df_as_latex)
            plt.savefig(os.path.join(img_dir,
                                     "top_" + str(top_n) + measure + indicator + ".png"), format="png", dpi=300)
            df_all = a2aa.get_mean_change_for_all_targets(groupby="to_stop_I", measure=measure)
            plot_df_point_data(df_all["mean_diff_mean"], df_all["lat"], df_all["lon"],
                               measure + "_winners_and_losers",
                               spatial_bounds=SPATIAL_BOUNDS_LM, img_dir=output_path,
                               color_map_params={"name": "seismic", "min": -2, "max": 2} if measure == "n_boardings" else
                               {"name": "seismic", "min": -30, "max": 30},
                               annotates=annotates)

       # for row in df.itertuples():
       #     df_as_latex.replace("value\_"+str(row.to_stop_I)+"\_to\_replace",
       #                         """\begin{figure}[H]
       #                         \includegraphics[width=\linewidth]{figs/diff_""" + str(row.to_stop_I) +
        #                        """-temporal_distance-diff_mean_relative.png}
        #                        \end{figure}""")

        #print(stops_remaining)



"""
plot_largest_stop_diff_component_maps(targets=[TARGET_DICT["soukka"]])



Select from_stop_I, N_stops from
(select from_stop_I, count(to_stop_I) as N_stops from temporal_distance WHERE mean >10800 AND from_stop_I not in (6723,) group by from_stop_I) q1
where N_stops = min(select count(to_stop_I) as N_stops from temporal_distance WHERE mean >10800 AND from_stop_I not in (6723,) group by from_stop_I)
"""


"""
-------------------------------------------NOT USED---------------------------------------------------------------------
These are attempts to do analyze and filter stops with o-d pairs where TT values are infinite
"""
if False:
    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, A2AA_DB_OLD_PATH, A2AA_DB_LM_PATH, A2AA_OUTPUT_DB_PATH)
    stop_list, ignore_list = a2aa.find_stops_where_all_indicators_are_finite(indicator="max")
    pickle.dump(stop_list, open("/home/clepe/production/results/helsinki/pickle_stops_to_include.pickle", 'wb'), -1)
    pickle.dump(ignore_list, open("/home/clepe/production/results/helsinki/pickle_stops_to_include.pickle", 'wb'), -1)

if False:
    # plot number of o-d pairs where measure is infinite
    a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, A2AA_DB_OLD_PATH, A2AA_DB_LM_PATH, A2AA_OUTPUT_DB_PATH)
    gtfs = GTFS(GTFS_PATH)
    for routing in ["after", "before"]:
        for measure in ["min", "max", "mean", "median"]:
            df = a2aa.n_inf_stops_per_stop("temporal_distance", measure, 10800, "to_stop_I", routing=routing)
            df = df.loc[df["N_stops"] >= 900]
            df = gtfs.add_coordinates_to_df(df, stop_id_column="to_stop_I", lat_name="lat", lon_name="lon")
            plot_df_point_data(df["N_stops"], df["lat"], df["lon"], "n_inf_in_"+measure+"_temp_dist_"+routing)

# not used
if False:
    measures = ["n_boardings", "temporal_distance"]
    indicators = ["diff_mean_relative", "diff_mean"]
    for measure in measures:
        for indicator in indicators:
            single_stop_change_histogram(0, measure, indicator=indicator, direction="to")

if False:
    for groupby in ["to_stop_I", "from_stop_I"]:
        for measure in ["n_boardings", "temporal_distance"]:
            if measure == "n_boardings":
                columns = ["decr_count_over_1_5", "decr_count_over_1", "decr_count_over_0_5",
                           "incr_count_over_1_5", "incr_count_over_1", "incr_count_over_0_5"]
            else:
                columns = ["decr_count_over_20", "decr_count_over_10", "decr_count_over_5",
                           "incr_count_over_20", "incr_count_over_10", "incr_count_over_5"]
            for column in columns:
                a2a_change_map(groupby=groupby, measure=measure, column=column)