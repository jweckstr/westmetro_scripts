"""
Analysis zones:
- vicinity of new metro stations, defined by walking distance to station
- vicinity of old metro stations, --||--
- areas relying on feeder bus (west metro), areas not in vicinity of metro stations and with direct bus to Kamppi
- commuter train station vicinities
- other areas
"""

# TODO: implement feeder bus zones based on polygon
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib import colors as mcolors
import itertools
import numpy as np
import pickle

import gtfspy.smopy_plot_helper
from gtfspy.gtfs import GTFS
from gtfspy.util import difference_of_pandas_dfs, makedirs
from gtfspy.mapviz_using_smopy_helper import plot_stops_with_categorical_attributes
from research.westmetro_paper.scripts.all_to_all_settings import *
from research.westmetro_paper.scripts.all_to_all_analyzer import AllToAllDifferenceAnalyzer
from gtfspy.routing.journey_data_analyzer import JourneyDataAnalyzer
from research.westmetro_paper.scripts.all_to_all_analyzer import stops_to_exclude

# ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]
zone_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]


def analysis_zones(as_dict=False):
    """
    returns data containers that pair zone type to a set of stops
    :param as_dict:
    :return:
    """
    gtfs_old = GTFS(OLD_DICT["gtfs_dir"])
    gtfs_lm = GTFS(LM_DICT["gtfs_dir"])
    station_distance = 600
    upstream_ratio = 0.5
    df_old = gtfs_old.get_stops_for_route_type(1)
    df_lm = gtfs_lm.get_stops_for_route_type(1)
    new_metro = difference_of_pandas_dfs(df_old, df_lm, ["stop_I"])
    old_metro = difference_of_pandas_dfs(new_metro, df_lm, ["stop_I"])
    train = gtfs_lm.get_stops_for_route_type(2)
    feeder_area = pd.DataFrame()
    other_stops = gtfs_lm.stops()
    jda = JourneyDataAnalyzer(LM_DICT["journey_dir"], LM_DICT["gtfs_dir"])
    # jda = JourneyDataAnalyzer(OLD_DICT["journey_dir"], OLD_DICT["gtfs_dir"])

    areas_to_remove = stops_to_exclude(return_sqlite_list=False)
    df = jda.get_upstream_stops_ratio(1040, [str(i.stop_I) for i in new_metro.itertuples()], upstream_ratio)
    feeder_area = feeder_area.append(df)
    # df = jda.get_upstream_stops_ratio(7193, 563, 0.7)
    print("new metro")
    for i in new_metro.itertuples():
        df = gtfs_lm.get_stops_within_distance(i.stop_I, station_distance)
        new_metro = new_metro.append(df)

    print("old metro")

    for i in old_metro.itertuples():
        df = gtfs_lm.get_stops_within_distance(i.stop_I, station_distance)
        old_metro = old_metro.append(df)
    print("train")

    for i in train.itertuples():
        df = gtfs_lm.get_stops_within_distance(i.stop_I, station_distance)
        train = train.append(df)

    new_metro = new_metro.drop_duplicates().reset_index(drop=True)
    old_metro = old_metro.drop_duplicates().reset_index(drop=True)
    train = train.drop_duplicates().reset_index(drop=True)
    feeder_area = feeder_area.drop_duplicates().reset_index(drop=True)

    # cleaning up borders
    new_metro = difference_of_pandas_dfs(old_metro, new_metro, ["stop_I"])
    for zone in [new_metro, old_metro, areas_to_remove]:
        train = difference_of_pandas_dfs(zone, train, ["stop_I"])
    for zone in [new_metro, train, old_metro, areas_to_remove]:
        feeder_area = difference_of_pandas_dfs(zone, feeder_area, ["stop_I"])

    spec_areas = pd.concat([new_metro, old_metro, train, feeder_area, areas_to_remove])

    other_stops = difference_of_pandas_dfs(spec_areas, other_stops, ["stop_I"])

    old_metro = old_metro.assign(stop_cat=1)
    new_metro = new_metro.assign(stop_cat=2)
    train = train.assign(stop_cat=3)
    feeder_area = feeder_area.assign(stop_cat=4)
    other_stops = other_stops.assign(stop_cat=5)
    all_stops = pd.concat([new_metro, old_metro, train, feeder_area, other_stops]).reset_index(drop=True)
    if as_dict:
        all_dfs = {"new_metro_stations": new_metro,
                   "feeder_bus_area": feeder_area,
                   "old_metro_stations": old_metro,
                   "commuter_train_stations": train,
                   "other_stops": other_stops}
    else:
        all_dfs = [("new_metro_stations", new_metro),
                   ("feeder_bus_area", feeder_area),
                   ("old_metro_stations", old_metro),
                   ("commuter_train_stations", train),
                   ("other_stops", other_stops)]
    return all_dfs, all_stops


def zone_map(img_dir=None, targets=True):
    # zone map
    all_dfs, _ = analysis_zones()
    all_lats = [x[1]["lat"] for x in reversed(all_dfs)]
    all_lons = [x[1]["lon"] for x in reversed(all_dfs)]
    all_cats = [x[1]["stop_cat"] for x in reversed(all_dfs)]
    all_labels = [x[0].replace("_", " ") for x in reversed(all_dfs)]
    #fig = plt.figure()

    ax = plot_stops_with_categorical_attributes(all_lats, all_lons, all_cats, labels=all_labels,
                                                                  spatial_bounds=SPATIAL_BOUNDS,
                                                                  colors=zone_colors,
                                                                  s=20)
    #print(list((x[0] for x in reversed(all_dfs))))

    ax.legend(scatterpoints=1,
              loc='upper left',
              ncol=1,
              fontsize=8)
    """,
               #(x[0] for x in reversed(all_dfs)),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=8)
"""
    gtfs_old = GTFS(OLD_DICT["gtfs_dir"])
    """
    for name, stop_I in TARGET_DICT.items():
        lat, lon = gtfs_old.get_stop_coordinates(stop_I)
        #ax.scatter(lon, lat, s=30, c='green', marker='X')
        ax.text(lon, lat, TARGET_LETTERS[name], size=7, color='black') #,
                #bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 0})
    """
    if not img_dir:
        img_dir = "/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"
    plt.savefig(os.path.join(img_dir,
                             "study_areas"+".pdf"), format="pdf", dpi=300, bbox_inches='tight')


def get_combinations(a2aa, measure="mean", mode="temporal_distance", rerun=True, unit="s"):
    """
    Returns rows for each combination of zone type
    :param a2aa:
    :param measure:
    :param rerun:
    :param mode:
    :param unit:
    :return:
    """
    all_dfs, _ = analysis_zones()
    combinations = itertools.product(all_dfs, all_dfs)
    dfs = {}
    pickle_path = os.path.join(makedirs("/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"),
                               "dataframe.pickle")

    if rerun:
        for ((i_name, i), (j_name, j)) in combinations:
            dfs[(i_name, j_name)] = a2aa.get_rows_based_on_stop_list(i["stop_I"], j["stop_I"], measure, mode, unit=unit)

        pickle.dump(dfs, open(pickle_path, 'wb'), -1)
    else:
        dfs = pickle.load(open(pickle_path, 'rb'))
    return combinations, dfs, all_dfs


def get_combinations_and_to_all(a2aa, measure="mean", mode="temporal_distance", rerun=True, unit="s"):
    """
    Returns rows for each combination of zone type
    :param a2aa:
    :param measure:
    :param rerun:
    :param mode:
    :param unit:
    :return:
    """
    all_dfs, all_stops = analysis_zones()
    row_dfs = all_dfs
    col_dfs = [("all_stops", all_stops)] + all_dfs
    combinations = itertools.product(row_dfs, col_dfs)
    #print([(c[0][0],c[1][0]) for c in combinations])
    dfs = {}
    pickle_path = os.path.join(makedirs("/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"),
                               mode+"_c_and_all_dataframe.pickle")

    if rerun:
        for ((i_name, i), (j_name, j)) in combinations:
            dfs[(i_name, j_name)] = a2aa.get_rows_based_on_stop_list(i["stop_I"], j["stop_I"], measure, mode, unit=unit)

        pickle.dump(dfs, open(pickle_path, 'wb'), -1)
    else:
        dfs = pickle.load(open(pickle_path, 'rb'))
    return combinations, dfs, row_dfs, col_dfs


def get_zone_to_all(a2aa, measure_mode, measure="mean", rerun=True):
    """
    Returns rows for each combination of zone type
    :param a2aa:
    :param measure:
    :param rerun:
    :return:
    """
    all_dfs, all_stops = analysis_zones()

    dfs = {}
    pickle_path = os.path.join(makedirs("/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"),
                               measure_mode+"_z2a_dataframe.pickle")

    if rerun:
        for ba in ["before", "after"]:
            for (i_name, i) in all_dfs:
                dfs[(i_name, ba)] = a2aa.get_rows_based_on_stop_list(i["stop_I"], all_stops["stop_I"], measure,
                                                                     measure_mode, unit="s")

        pickle.dump(dfs, open(pickle_path, 'wb'), -1)
    else:
        dfs = pickle.load(open(pickle_path, 'rb'))
    return dfs, all_dfs


def old_vs_change_scatter_colored_by_zone(a2aa, measure="temporal_distance", file_id=None, img_dir=None):
    all_dfs, _ = analysis_zones()
    all_stops = []
    #for df in all_dfs:
    #    all_stops += df[1]["stop_I"].tolist()
    fig = plt.figure()

    plt.title("", fontsize=20)
    ax = fig.add_subplot(111)
    for (name, df), c in zip(reversed(all_dfs), zone_colors):
        df = a2aa.get_mean_change(measure, include_list=df["stop_I"])
        ax.scatter(df["before"].apply(lambda x: x/60), df["diff_mean"].apply(lambda x: x/60), label=name, c=c, alpha=0.1)
    ax.set_xlim([40, 120])
    ax.set_ylim([-20, 20])
    ax.legend()
    plt.xlabel("Before MTT, minutes")
    plt.ylabel("Change in MTT, minutes")
    if not img_dir:
        img_dir = "/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"
    if not file_id:
        file_id = ''
    plt.savefig(os.path.join(img_dir,
                             "before_vs_change_scatter_"+str(file_id)+".pdf"), format="pdf", dpi=300)


def heatmap_singles(a2aa, measure="mean", rerun=True, img_dir=None):
    # heatmap single
    combinations, dfs, all_dfs = get_combinations(a2aa, measure=measure, rerun=rerun)
    for ((i_name, i), (j_name, j)) in combinations:
        df = dfs[(i_name, j_name)]
        print(i_name, j_name)
        xedges = range(0, 90, 1)
        yedges = range(-30, 30, 1)
        H, xedges, yedges = np.histogram2d(df["before_"+measure], df["diff_"+measure], bins=(xedges, yedges))
        H = H.T  # Let each row list bins with common y range.
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, title=i_name+" to "+j_name)
        # ax.plot([0, 90], [0, 90], c="r")
        plt.xlabel("before " + measure + " temporal distance (s)")
        plt.ylabel("after-before " + measure + " temporal distance (s)")
        plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        if not img_dir:
            img_dir = "/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"
        plt.savefig(os.path.join(img_dir,
                                 "diff_"+i_name+"-"+j_name+".pdf"), format="pdf", dpi=300)


def heatmap_matrix(a2aa, measure="mean", mode="temporal_distance", rerun=True, img_dir=None, fig_name=None):
    # heatmap matrix
    combinations, dfs, all_dfs = get_combinations(a2aa, measure=measure, mode=mode, rerun=rerun, unit="s")

    width = 10
    height = 8
    fig, axes2d = plt.subplots(nrows=len(all_dfs), ncols=len(all_dfs), figsize=(width, height))
    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):

            i_name, i_df = all_dfs[i]
            j_name, j_df = all_dfs[j]
            df = dfs[(i_name, j_name)]
            print(i_name, j_name)
            if mode == "temporal_distance":
                xedges = range(0, 90, 1)
                yedges = range(-30, 30, 1)
            else:
                xedges = range(0, 4, 1)
                yedges = range(-2, 2, 1)
            H, xedges, yedges = np.histogram2d(df["before_"+measure], df["diff_"+measure], bins=(xedges, yedges))
            #H = H.T  # Let each row list bins with common y range.
            #plt.xlabel("before " + measure + " temporal distance (s)")
            #plt.ylabel("after-before " + measure + " temporal distance (s)")
            cell.imshow(H.T, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            cell.set_xlim(min(xedges), max(xedges))
            cell.set_ylim(min(yedges), max(yedges))
            cell.yaxis.tick_right()
            # cell.xaxis.tick_top()
            cell.xaxis.set_label_position('top')

            if i == len(axes2d) - 5:
                cell.set_xlabel(j_name)
            if not i == len(axes2d)-1:
                cell.set_xticks([])
            if j == 0:
                cell.set_ylabel(i_name)
            if not j == 4:
                cell.set_yticks([])

    fig.text(0.5, 0.04, "Travel time, before (minutes)", ha='center')
    fig.text(0.04, 0.5, "Difference in travel time (minutes)", va='center', rotation='vertical')
    fig.text(0.5, 1-0.04, "To", ha='center')
    fig.text(1-0.04, 0.5, "From", va='center', rotation=-90)
    fig.tight_layout()
    if not img_dir:
        img_dir = "/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"
    plt.savefig(os.path.join(img_dir,
                             "diff_heatmap_matrix" +fig_name+ ".png"), format="png", dpi=300)


def histogram_matrix(a2aa, rerun, measure="mean", mode="temporal_distance", img_dir=None, fig_name=None):
    # histogram matrix
    if mode == "temporal_distance":
        yedges = range(-20, 20, 1)
        unit = "m"
    else:
        yedges = np.linspace(-3, 3, 40)
        unit = "s"
    combinations, dfs, row_dfs, col_dfs = get_combinations_and_to_all(a2aa, measure=measure, mode=mode, rerun=rerun,
                                                                      unit=unit)

    fig, axes2d = plt.subplots(nrows=len(row_dfs), ncols=len(col_dfs) + 1, figsize=(10, 8),
                               gridspec_kw={'width_ratios': [1, 0.5, 1, 1, 1, 1, 1]})
    fig.subplots_adjust(hspace=0.04, wspace=0.04)

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            if j == 1:
                cell.remove()
                continue
            if j > 1:
                j = j - 1
            i_name, _ = row_dfs[i]
            j_name, _ = col_dfs[j]
            print(dfs.keys())
            df = dfs[(i_name, j_name)]
            print(i_name, j_name)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()


            cell.yaxis.tick_right()

            n, bins, patches = cell.hist(np.array(df["diff_" + measure]), bins=yedges, normed=True, facecolor='green', alpha=0.75)
            cell.yaxis.tick_right()
            # cell.xaxis.tick_top()
            cell.xaxis.set_label_position('top')

            if i == 0:
                j_name = j_name.replace("_", " ")
                index = j_name.rfind(" ")
                j_name = j_name[:index] + '\n' + j_name[index:]
                cell.set_xlabel(j_name)
            if mode == "temporal_distance":
                cell.set_xticks([-15, 0, 15])
                cell.set_ylim(0, 0.2)
                cell.set_yticks([0.0, 0.05, 0.10, 0.15])
            else:
                cell.set_xticks([-2, 0, 2])
                cell.set_ylim(0, 0.8)
                cell.set_yticks([0.0, 0.20, 0.40, 0.60])
            if not i == len(axes2d)-1:
                cell.set_xticklabels([])
            if j == 0:
                i_name = i_name.replace("_", " ", 1)
                #index = i_name.rfind(" ")
                #i_name = i_name[:index] + '\n' + j_name[index:]
                i_name = i_name.replace("_", '\n', 1)

                cell.set_ylabel(i_name)
            if not j == 5:
                cell.set_yticklabels([])

            """
            if i == len(axes2d) - 1:
                cell.set_xlabel(j_name.replace("_", " "), wrap=True)
            if j == 0:
                cell.set_ylabel(i_name.replace("_", " "), wrap=True)
            """
    #fig.tight_layout()
    if mode == "temporal_distance":
        fig.text(0.5, 0.04, "Change in travel time (minutes)", ha='center')
    else:
        fig.text(0.5, 0.04, "Change in number of transfers", ha='center')

    fig.text(0.04, 0.5, "From", va='center', rotation='vertical')
    fig.text(0.5, 1-0.04, "To", ha='center')
    #fig.text(1-0.01, 0.5, "From", va='center', rotation=-90, bbox={'facecolor': 'white', 'pad': 0.5})
    #
    if not img_dir:
        img_dir = "/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"
    plt.savefig(os.path.join(img_dir,
                             "diff_histogram_matrix" +fig_name+ ".pdf"), format="pdf")#, dpi=300, bbox_inches='tight')


def ba_histogram_matrix(a2aa, rerun, measure_mode, measure="mean", img_dir=None, fig_name=None):
    # before after histogram matrix
    assert measure_mode in ["n_boardings", "temporal_distance"]
    dfs, all_dfs = get_zone_to_all(a2aa, measure_mode, measure=measure, rerun=rerun)
    ba = ["before", "after"]
    colors = ['black', 'red']
    fig, axes2d = plt.subplots(nrows=len(all_dfs), ncols=1, figsize=(5, 10))
    #plt.rcParams.update({'font.size': 22})

    #plt.rcParams['svg.fonttype'] = 'none'
    font_size = 22
    for i, cell in enumerate(axes2d):
        for j, j_name in enumerate(ba):

            i_name, _ = all_dfs[i]
            df = dfs[(i_name, j_name)]
            print(i_name, j_name)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            #print(df)
            if measure_mode == "temporal_distance":
                bins = range(0, 120, 1)
            else:
                bins = np.linspace(0, 5, 120)
            n, bins, patches = cell.hist(np.array(df[j_name+"_" + measure]), bins=bins, normed=True,
                                         color=colors[j], alpha=1, histtype='step')

            if measure_mode == "temporal_distance":
                cell.set_ylim(0, 0.03)
            else:
                cell.set_ylim(0, 6)
            if i == len(axes2d) - 1:
                if measure_mode == "temporal_distance":
                    cell.set_xlabel("minutes", wrap=True)
                else:
                    cell.set_xlabel("transfers", wrap=True)
                #cell.xaxis.label.set_size(font_size)
            cell.set_ylabel(i_name.replace("_", " "), wrap=True)#, fontsize=font_size)
            #cell.yaxis.label.set_size(font_size)

    labels = ba
    handles = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles, labels)

    fig.tight_layout()
    if not img_dir:
        img_dir = "/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps"
    plt.savefig(os.path.join(img_dir,
                             "diff_histogram_matrix" +fig_name+ ".pdf"), format="pdf")#, dpi=300)


def distance_vs_rows_histogram(a2aa, img_dir=None):
    ignore_stops = stops_to_exclude(return_sqlite_list=True)
    measure = "mean"
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    n_value = 180
    for n, sign in zip([-1*n_value, n_value], ["<=", ">="]):
        df = a2aa.get_rows_with_abs_change_greater_than_n(ignore_stops, measure, n, sign, unit="s")
        n, bins, patches = ax.hist(np.array(df["before_"+measure]), normed=True, facecolor='green', alpha=0.75)
        plt.ylim(0, 0.2)


    plt.xlabel("travel time")
    plt.ylabel("number of stop_pairs")
    if not img_dir:
        img_dir = makedirs("/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps")
    plt.savefig(os.path.join(img_dir,
                             "distance_vs_volume_of_change_" + str(n) + ".png"), format="png", dpi=300)


def single_stop_change_histogram(target, measure, direction="to", indicator="diff_mean", a2aa=None, img_dir=None, ax=None, return_ax=False, cdf=False, color='blue', label=''):
    if not a2aa:
        a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, A2AA_DB_OLD_PATH, A2AA_DB_LM_PATH, A2AA_OUTPUT_DB_PATH)
    if not ax:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, title="")

    if measure == "n_boardings":
        yedges = np.arange(-2.0, 2.0, 0.1)
        unit = "s"
    else:
        yedges = range(-25, 25, 1)
        unit = "m"
    if indicator == "diff_mean_relative":
        yedges = np.arange(-0.7, 0.7, 0.05)
        unit = "s"

    df = a2aa.get_data_for_target(target, measure, direction=direction, unit=unit, ignore_stops=True)
    if cdf:
        values, base = np.histogram(np.array(df[indicator]), bins=yedges)
        # evaluate the cumulative
        cumulative = np.cumsum(values)
        # plot the cumulative function
        ax.plot(base[:-1], cumulative, c=color, label=label)
        plt.ylim(0, max(cumulative))

    else:
        n, bins, patches = ax.hist(np.array(df[indicator]), bins=yedges, normed=True, facecolor='green', alpha=0.75)
        plt.ylim(0, 0.2)

    # ax.plot([0, 90], [0, 90], c="r")
    if return_ax:
        return ax
    plt.xlabel("")
    plt.ylabel("")
    if not img_dir:
        img_dir = makedirs("/home/clepe/production/results/helsinki/figs/all_to_all/heatmaps")
    plt.savefig(os.path.join(img_dir,
                             "diff_" + str(target) + "-" + measure + "-" + indicator + ".pdf"), format="pdf", dpi=300)


if __name__ == "__main__":
    zone_map()

