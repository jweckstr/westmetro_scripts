import os
import pytz

CITY_NAME = "helsinki"
RESULTS_DIRECTORY = os.path.join("/home/clepe/results", CITY_NAME)
FIGS_DIRECTORY = "../figs/"


def shapefile_dir(name):
    SHAPEFILE_DIR = os.path.join(RESULTS_DIRECTORY, CITY_NAME+"_"+name+".shp")
    return SHAPEFILE_DIR

GTFS_DATA_BASEDIR = "/home/clepe/scratch/merged_feeds/"+CITY_NAME+"/2016-12-07/daily.sqlite"
NODES_FNAME = GTFS_DATA_BASEDIR + "daily.nodes.csv"
# "main.day.nodes.csv"
TRANSIT_CONNECTIONS_FNAME = os.path.join(GTFS_DATA_BASEDIR, "daily.temporal_network.csv")
JOURNEY_DATA_DIR = os.path.join(RESULTS_DIRECTORY, CITY_NAME+"_test.sqlite")
DEFAULT_TILES = "CartoDB positron"
DARK_TILES = "CartoDB dark_matter"

DAY_START = 1475438400 + 3600
DAY_END = DAY_START + 24 * 3600

ROUTING_START_TIME_DEP = DAY_START + 8 * 3600  # 07:00 AM

ANALYSIS_START_TIME_DEP = ROUTING_START_TIME_DEP
ANALYSIS_END_TIME_DEP = ROUTING_START_TIME_DEP + 1 * 3600

ROUTING_END_TIME_DEP = ROUTING_START_TIME_DEP + 3 * 3600

TARGET_STOPS = [4069]
WALK_SPEED = 70.0 / 60
TRANSFER_MARGIN = 180

# TIMEZONE = pytz.timezone("Europe/Copenhagen")

ROUTING_PARAMS = {"day_start": DAY_START,
                  "day_end": DAY_END,
                  "routing_start_time_dep": ROUTING_START_TIME_DEP,
                  "routing_end_time_dep": ROUTING_END_TIME_DEP,
                  "analysis_start_time_dep": ANALYSIS_START_TIME_DEP,
                  "analysis_end_time_dep": ANALYSIS_END_TIME_DEP,
                  "walking_speed": WALK_SPEED,
                  "transfer_margin": TRANSFER_MARGIN}