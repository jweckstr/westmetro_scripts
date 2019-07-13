import os

DEMAND_ZONES = "/home/clepe/gis/transit/demand_matrix/zones/hsl_zones_wgs84.shp"

CITY_NAME = "helsinki"
OSM_DIR = "research/westmetro_paper/osm/helsinki_finland.osm.pbf"
GTFS_DB_SOURCE_DIR = "research/westmetro_paper/data/helsinki/backups"

GTFS_DB_WORK_DIR = "research/westmetro_paper/data/helsinki"
GTFS_DB_OLD = "old_daily"
GTFS_DB_LM = "lm_daily"

GTFS_FEEDS = [GTFS_DB_OLD, GTFS_DB_LM]

SQLITE_SUFFIX = ".sqlite"
RESULTS_PREFIX = "results_"
PSEUDO_STOP_FNAME = "pseudo_stops.csv"
DIFF_DB_FNAME = "diff"
TRAVEL_IMPEDANCE_STORE_FNAME = "travel_impedance_store"
RESULTS_DIR = "research/westmetro_paper/results/helsinki"
PICKLE_PREFIX = "pickle_"
PICKLE = True
DIFF_PATH = os.path.join(RESULTS_DIR, DIFF_DB_FNAME+SQLITE_SUFFIX)
TRAVEL_IMPEDANCE_STORE_PATH = os.path.join(RESULTS_DIR, TRAVEL_IMPEDANCE_STORE_FNAME+SQLITE_SUFFIX)

GTFS_DB = GTFS_DB_LM
OTHER_DB = GTFS_DB_OLD if GTFS_DB == GTFS_DB_LM else GTFS_DB_LM
GTFS_DB_FNAME = GTFS_DB+SQLITE_SUFFIX
OTHER_DB_FNAME = OTHER_DB+SQLITE_SUFFIX
JOURNEY_DB_FNAME = RESULTS_PREFIX+GTFS_DB_FNAME
OTHER_JOURNEY_DB_FNAME = RESULTS_PREFIX+OTHER_DB_FNAME

A2AA_DB_LM = "travel_impedance_measures_lm_{time}.sqlite"
A2AA_DB_OLD = "travel_impedance_measures_old_{time}.sqlite"
A2AA_OUTPUT_DB = "a2a_output{time}.sqlite"

A2AA_DB_LM_PATH = os.path.join(RESULTS_DIR, A2AA_DB_LM)
A2AA_DB_OLD_PATH = os.path.join(RESULTS_DIR, A2AA_DB_OLD)
A2AA_OUTPUT_DB_PATH = os.path.join(RESULTS_DIR, A2AA_OUTPUT_DB)
GTFS_PATH = os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_FNAME)


def get_a2aa_db_path(time, type):
    if type == "lm":
        return os.path.join(RESULTS_DIR, A2AA_DB_LM.format(time=time))
    elif type == "old":
        return os.path.join(RESULTS_DIR, A2AA_DB_OLD.format(time=time))
    elif type == "output":
        return os.path.join(RESULTS_DIR, A2AA_OUTPUT_DB.format(time=time))


FIGS_DIRECTORY = os.path.join(RESULTS_DIR, "figs")

# FEED SETTINGS:
CUTOFF_DISTANCE = 1000

# ROUTING PARAMETERS:
TRANSFER_MARGIN = 180
WALK_SPEED = 70.0 / 60

TRACK_ROUTE = True

TARGET_DICT = {"Rautatientori": 1130, "Kamppi": 1032, "Lasipalatsi": 1118, "Soukka": 382, "Matinkyla": 543, "Otaniemi": 1711}
TARGET_LETTERS = {"Rautatientori": "C", "Kamppi": "A", "Lasipalatsi": "B", "Soukka": "D", "Matinkyla": "E", "Otaniemi": "F"}
"""{"Leppävaara": 3235,
               "Kamppi": 1032,
               "Rautatientori": 1130,
               "Airport": 6604,
               "HYKS": 1897,
               "Itäkeskus": 2732,
               "Pohjois-Tapiola, mall": 1719,
               "Matinkylä, south": 543,
               "Olari, center": 1432,
               "Soukka, center": 382}"""
# , "pasila": 7249,  "herttoniemi": 7417,  "matinkylä": 7472,  "soukka": 7495, "tikkurila": 7678
TARGET_STOPS = TARGET_DICT.values()

if TRACK_ROUTE:
    TRACK_VEHICLE_LEGS = True
    TRACK_TIME = True

else:
    TRACK_VEHICLE_LEGS = True
    TRACK_TIME = True

ROUTING_START_TIME_DS = 7 * 3600
ROUTING_END_TIME_DS = 10 * 3600
ANALYSIS_START_TIME_DS = 7 * 3600
ANALYSIS_END_TIME_DS = 8 * 3600

#SPATIAL_BOUNDS = {'lat_max': 60.444539, "lon_max": 25.315532, "lat_min": 60.067231, "lon_min": 24.372292}

SPATIAL_BOUNDS = {'lat_max': 60.370740, "lon_max": 25.200537, "lat_min": 60.113027, "lon_min": 24.528014}

SPATIAL_BOUNDS_LM = {'lat_max': 60.213273, "lon_max": 24.903883, "lat_min": 60.099057, "lon_min": 24.580504}

TIMES = [7, 12, 16, 21]

def get_feed_dict(gtfs_fname, day_start, feed_seq):
    return {"gtfs_dir": os.path.join(GTFS_DB_WORK_DIR, gtfs_fname+SQLITE_SUFFIX),
            "journey_dir": os.path.join(RESULTS_DIR, RESULTS_PREFIX+gtfs_fname+SQLITE_SUFFIX),
            "pickle_dir": os.path.join(RESULTS_DIR, PICKLE_PREFIX+gtfs_fname),
            "day_start": day_start,
            "day_end": day_start+24*3600,
            "routing_start_time": day_start + ROUTING_START_TIME_DS,
            "routing_end_time": day_start + ROUTING_END_TIME_DS,
            "analysis_start_time": day_start + ANALYSIS_START_TIME_DS,
            "analysis_end_time": day_start + ANALYSIS_END_TIME_DS,
            "feed_seq": feed_seq,
            "feed_desc": "Before" if gtfs_fname == "old_daily" else "After"
            }

LM_DICT = get_feed_dict("lm_daily", 1515535200, 1)

OLD_DICT = get_feed_dict("old_daily", 1508878800, 0)

FEED_LIST = [("old_daily", OLD_DICT), ("lm_daily", LM_DICT)]

FEED_DICT = {"old_daily": OLD_DICT, "lm_daily": LM_DICT}


def routing_params(feed_name):
    return {"day_start": FEED_DICT[feed_name]["day_start"],
            "day_end": FEED_DICT[feed_name]["day_end"],
            "routing_start_time_dep": FEED_DICT[feed_name]["routing_start_time"],
            "routing_end_time_dep": FEED_DICT[feed_name]["routing_end_time"],
            "walking_speed": WALK_SPEED,
            "walk_speed": WALK_SPEED,
            "transfer_margin": TRANSFER_MARGIN,
            "track_vehicle_legs": TRACK_VEHICLE_LEGS}
