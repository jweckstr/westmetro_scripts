"""
Start with before, after DB's
homogenize_stops_table_with_other_db
replace_stop_i_with_stop_pair_i

"""
import geopandas as gpd
from shapely.geometry import Point
from scripts.all_to_all_settings import *
from gtfspy.gtfs import GTFS
from sqlite3 import OperationalError


def add_zone_to_stop_table(zone_shape_path=DEMAND_ZONES):
    """
    Creates table which relates stop_Is with TAZ zones and counts the number of stops
    :return:
    """
    crs = {"init": "espg:4326"}
    zones = gpd.read_file(zone_shape_path, crs=crs)
    for (name, gtfs_dict) in FEED_LIST:
        gtfs = GTFS(gtfs_dict["gtfs_dir"])
        df = gtfs.stops()
        geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
        df = df.drop(["lon", "lat"], axis=1)

        gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        zones_and_stops = gpd.sjoin(gdf, zones, how="inner", op='intersects')
        try:
            gtfs.execute_custom_query("""ALTER TABLE stops ADD COLUMN n_stops INT;""")
            gtfs.execute_custom_query("""ALTER TABLE stops ADD COLUMN zone_id INT;""")
        except OperationalError:
            pass
        subset = zones_and_stops[['WSP_ENN', 'stop_I']]
        tuples = [tuple(x) for x in subset.values]
        gtfs.conn.executemany("""UPDATE stops SET zone_id = ? WHERE stop_I = ?""", tuples)
        gtfs.conn.commit()
#"""CREATE TABLE zones_and_stops AS (stop_I INT, WSP_ENN INT, WSP_SIJ INT, n_stops INT)"""
add_zone_to_stop_table()