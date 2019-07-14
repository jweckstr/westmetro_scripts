import os
import sys
from flask import Flask
from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler
from gtfspy.gtfs import GTFS
from gtfspy.util import timeit
from gtfspy.networks import walk_transfer_stop_to_stop_network
from gtfspy.util import makedirs
from gtfspy.route_types import ROUTE_TYPE_TO_SHORT_DESCRIPTION
from scripts.all_to_all_analyzer import stops_to_exclude

from scripts.all_to_all_settings import *

if __name__ == "__main__":
    master_df = None
    ignore_stops = stops_to_exclude(return_sqlite_list=True)
    for feed, feed_dict in FEED_LIST:
        g = GTFS(feed_dict["gtfs_dir"])
        stats = g.get_stats()


        n_stops = g.execute_custom_query_pandas("""SELECT TYPE, count(*) AS n_stops FROM (SELECT * FROM stop_times, 
                                            stops, trips, routes 
                                            WHERE stops.stop_I=stop_times.stop_I AND trips.trip_I=stop_times.trip_I AND stops.stop_I NOT IN {ignore_stops}
                                            AND trips.route_I=routes.route_I
                                            GROUP BY stops.self_or_parent_I, type) q1
                                            GROUP BY type""".format(ignore_stops=ignore_stops))

        vehicle_kms = g.execute_custom_query_pandas("""SELECT type, sum(distance)/1000 AS vehicle_kilometers FROM
    (SELECT find_distance(q1.lon, q1.lat, q2.lon, q2.lat) AS distance, q1.stop_I AS from_stop_I, q2.stop_I AS to_stop_I, 
    type FROM
    (SELECT * FROM stop_times, stops, trips, routes WHERE stops.stop_I=stop_times.stop_I 
    AND trips.trip_I=stop_times.trip_I AND trips.route_I=routes.route_I  AND stops.stop_I NOT IN {ignore_stops}) q1,
    (SELECT * FROM stop_times, stops WHERE stops.stop_I=stop_times.stop_I AND stops.stop_I NOT IN {ignore_stops}) q2
    WHERE q1.seq+1 = q2.seq AND q1.trip_I = q2.trip_I) sq1
    GROUP BY type""".format(ignore_stops=ignore_stops))

        network_length = g.execute_custom_query_pandas("""SELECT type, sum(distance)/1000 AS route_kilometers FROM
    (SELECT find_distance(q1.lon, q1.lat, q2.lon, q2.lat) AS distance, q1.stop_I AS from_stop_I, q2.stop_I AS to_stop_I, 
    type FROM
    (SELECT * FROM stop_times, stops, trips, routes 
    WHERE stops.stop_I=stop_times.stop_I AND trips.trip_I=stop_times.trip_I AND trips.route_I=routes.route_I  
    AND stops.stop_I NOT IN {ignore_stops}) q1,
    (SELECT * FROM stop_times, stops 
    WHERE stops.stop_I=stop_times.stop_I AND stops.stop_I NOT IN {ignore_stops}) q2
    WHERE q1.seq+1 = q2.seq and q1.trip_I = q2.trip_I
    GROUP BY from_stop_I, to_stop_I) sq1
    GROUP BY type """.format(ignore_stops=ignore_stops))
        #print(n_stops)

        #print(network_length)
        #print(vehicle_kms)
        if master_df is None:
            master_df = n_stops
            for table in [vehicle_kms, network_length]:
                master_df = master_df.join(table.loc[:, table.columns != 'type'], how="left")
        else:
            for table in [n_stops, vehicle_kms, network_length]:
                master_df = master_df.join(table.loc[:, table.columns != 'type'],
                                           how="left", lsuffix="_before", rsuffix="_after")
    master_df = master_df.astype(int)
    master_df["mode"] = master_df["type"].apply(lambda x: ROUTE_TYPE_TO_SHORT_DESCRIPTION[x])

    master_df = master_df.round(0)
    master_df = master_df.append(master_df.agg(['sum']))

    print(master_df[["mode", "n_stops_before", "n_stops_after", "route_kilometers_before",
                     "route_kilometers_after", "vehicle_kilometers_before", "vehicle_kilometers_after"]].to_latex(index=False))
