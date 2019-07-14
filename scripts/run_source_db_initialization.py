import os
import subprocess
from gtfspy.gtfs import GTFS
from sqlite3 import IntegrityError

from gtfspy.osm_transfers import compute_stop_to_stop_osm_walk_distances_python
from scripts.all_to_all_settings import *
from gtfspy.filter import FilterExtract
from gtfspy.import_gtfs import import_gtfs
from gtfspy.import_validator import ImportValidator
from gtfspy.aggregate_stops import aggregate_stops_spatially, merge_stops_tables

names = ['old', 'lm']
dates = ['2017-10-25', '2018-01-10']
gtfs_lm_zip = "hsl_20180105T215101Z.zip"
gtfs_a17_zip = 'hsl_20171018T005101Z.zip'

# imports: python3 import_gtfs.py import /home/clepe/production/data/helsinki/october_2017.zip /home/clepe/production/data/helsinki/old_all.sqlite
# filter:


def import_from_zips():
    for name, zip_path, date in zip(names, [gtfs_a17_zip, gtfs_lm_zip], dates):
        if name == 'old':
            continue
        import_gtfs(os.path.join(GTFS_DB_WORK_DIR, zip_path), os.path.join(GTFS_DB_WORK_DIR, name + '_all.sqlite'),
                    location_name='helsinki')
        gtfs = GTFS(os.path.join(GTFS_DB_WORK_DIR, name + '_all.sqlite'))

        f = FilterExtract(gtfs, os.path.join(GTFS_DB_WORK_DIR, name + '_daily.sqlite'), date=date)
        f.create_filtered_copy()
        gtfs = GTFS(os.path.join(GTFS_DB_WORK_DIR, name + '_all.sqlite'))
        iv = ImportValidator(os.path.join(GTFS_DB_WORK_DIR, zip_path), gtfs)
        warnings = iv.validate_and_get_warnings()
        warnings.write_summary()


def prepare_dbs():
    try:
        assert not os.path.isfile(os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_LM+SQLITE_SUFFIX))
        assert not os.path.isfile(os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_OLD+SQLITE_SUFFIX))

        for fn in [GTFS_DB_LM+SQLITE_SUFFIX, GTFS_DB_OLD+SQLITE_SUFFIX]:
            subprocess.call(["cp", os.path.join(GTFS_DB_SOURCE_DIR, fn), os.path.join(GTFS_DB_WORK_DIR, fn)])
    except AssertionError:
        print("remove old files to start from scratch, continuing...")

    G_lm = GTFS(os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_LM+SQLITE_SUFFIX))

    G_old = GTFS(os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_OLD+SQLITE_SUFFIX))
    """try:
        except IntegrityError:
        print("additional stops already added")"""
        #G_lm.add_stops_from_csv(os.path.join(GTFS_DB_SOURCE_DIR, PSEUDO_STOP_FNAME))
    merge_stops_tables(os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_OLD+SQLITE_SUFFIX),
                       os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_LM+SQLITE_SUFFIX))

    G_old.replace_stop_i_with_stop_pair_i(colname="stop_pair_I")
    G_lm.replace_stop_i_with_stop_pair_i(colname="stop_pair_I")

    lm_stops = G_lm.execute_custom_query_pandas("SELECT * FROM stops")
    old_stops = G_old.execute_custom_query_pandas("SELECT * FROM stops")
    lm_stops_set = set(lm_stops["stop_I"])
    old_stops_set = set(old_stops["stop_I"])
    print(lm_stops_set)
    print(old_stops_set)
    print("stops not in old:", lm_stops_set - old_stops_set)
    print("stops not in old:", old_stops_set - lm_stops_set)


    print("calculating stop distances for first feed")
    G_old.recalculate_stop_distances(2000, remove_old_table=True)


    print("calculating stop distances for second feed")
    G_lm.recalculate_stop_distances(2000, remove_old_table=True)


        #G_lm.homogenize_stops_table_with_other_db(os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_OLD + SQLITE_SUFFIX))

    print("run walk distance routing")
    """
    add_walk_distances_to_db_python(G_lm, OSM_DIR, cutoff_distance_m=CUTOFF_DISTANCE)
    add_walk_distances_to_db_python(G_old, OSM_DIR, cutoff_distance_m=CUTOFF_DISTANCE)
    """
    for fn in [GTFS_DB_LM + SQLITE_SUFFIX, GTFS_DB_OLD + SQLITE_SUFFIX]:
        subprocess.call(["java", "-jar",
                         "gtfspy/java_routing/target/transit_osm_routing-1.0-SNAPSHOT-jar-with-dependencies.jar",
                         "-u",
                         os.path.join(GTFS_DB_WORK_DIR, fn),
                         "-osm",
                         OSM_DIR,
                         "--tempDir", "/tmp/"])

"""
compute_stop_to_stop_osm_walk_distances_python(G_lm, OSM_DIR, cutoff_distance_m=CUTOFF_DISTANCE)
compute_stop_to_stop_osm_walk_distances_python(G_old, OSM_DIR, cutoff_distance_m=CUTOFF_DISTANCE)


for fn in [GTFS_DB_LM + SQLITE_SUFFIX, GTFS_DB_OLD + SQLITE_SUFFIX]:
    subprocess.call(["java", "-jar",
                     "gtfspy/java_routing/target/transit_osm_routing-1.0-SNAPSHOT-jar-with-dependencies.jar",
                     "-u",
                     os.path.join(GTFS_DB_WORK_DIR, fn),
                     "-osm",
                     OSM_DIR,
                     "--tempDir", "/tmp/"])
if False:
    G_old.replace_stop_i_with_stop_pair_i()
    G_lm.replace_stop_i_with_stop_pair_i()
"""

prepare_dbs()