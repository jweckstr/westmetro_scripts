import sqlite3
import pandas
import itertools
import networkx as nx
from gtfspy.gtfs import GTFS
from gtfspy.util import timeit

from scripts.all_to_all_settings import *


def attach_database(conn, other_db_path, name="other"):
    cur = conn.cursor()

    cur.execute("ATTACH '%s' AS '%s'" % (str(other_db_path), name))
    cur.execute("PRAGMA database_list")
    print("other database attached:", cur.fetchall())
    return conn

"""
AllToAllDifferenceAnalyzer calculates the difference between various summary statistics of temporal distance and number
of boardings, stores the values in a database and handles calls to this database.
"""


def stops_to_exclude(return_sqlite_list=False):
    gtfs_lm = GTFS(LM_DICT["gtfs_dir"])
    areas_to_remove = gtfs_lm.execute_custom_query_pandas(
        "SELECT *  FROM stops  WHERE  CASE WHEN substr(stop_id,1, 5) = '__b__' THEN CAST(substr(stop_id,6, 1) AS integer) ELSE CAST(substr(stop_id,1, 1) AS integer) END >4")
    if return_sqlite_list:
        return "(" + ",".join([str(x) for x in areas_to_remove["stop_I"].tolist()]) + ")"
    return areas_to_remove


class AllToAllDifferenceAnalyzer:
    def __init__(self, gtfs_path, before_db_path, after_db_path, output_db):
        self.gtfs = GTFS(gtfs_path)
        print(output_db)
        self._create_indecies(before_db_path)
        self._create_indecies(after_db_path)

        self.conn = sqlite3.connect(output_db)
        self.conn = attach_database(self.conn, before_db_path, name="before")
        self.conn = attach_database(self.conn, after_db_path, name="after")

    def _create_indecies(self, db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        for table in ["journey_duration", "n_boardings", "temporal_distance"]:
            query = """CREATE INDEX IF NOT EXISTS %s_from_stop_I_idx ON %s (from_stop_I);
                CREATE INDEX IF NOT EXISTS %s_to_stop_I_idx ON %s (to_stop_I);""" % (table, table, table, table)
        conn.commit()

    def diff_table(self, groupby="to_stop_I", measure="temporal_distance", ignore_stops=None):
        """
        Creates a table with the before-after difference of mean, min and max temporal distance or number of boardings
        on a stop to stop basis
        :return:
        """

        cur = self.conn.cursor()
        query = """DROP TABLE IF EXISTS diff_{groupby}_{measure}""".format(measure=measure, groupby=groupby)
        cur.execute(query)

        multiplier = 1
        first = 0.5
        second = 1
        third = 1.5
        threshold = 10800  # threshold for change in mean temporal distance
        if measure == "temporal_distance" or "journey_duration":
            multiplier = 60
            first = 5
            second = 10
            third = 20

        first_str = str(first).replace(".", "_")
        second_str = str(second).replace(".", "_")
        third_str = str(third).replace(".", "_")

        if ignore_stops:
            ignore_stops = " AND t1.to_stop_I NOT IN " + ignore_stops + " AND t1.from_stop_I NOT IN " + ignore_stops
        else:
            ignore_stops = ""
        query = """CREATE TABLE IF NOT EXISTS diff_{groupby}_{measure}  ({groupby} INT, min_diff_mean REAL, mean_diff_mean REAL, 
        max_diff_mean REAL, incr_count_over_{0} INT, incr_count_over_{1} INT, incr_count_over_{2} INT, 
        decr_count_over_{0} INT, decr_count_over_{1} INT, decr_count_over_{2} INT )
        """.format(first_str, second_str, third_str,
                   measure=measure, groupby=groupby)
        cur.execute(query)

        query = """INSERT OR REPLACE INTO diff_{groupby}_{measure} ({groupby}, min_diff_mean, mean_diff_mean, max_diff_mean, 
                    incr_count_over_{first_str}, incr_count_over_{second_str}, incr_count_over_{third_str}, 
                    decr_count_over_{first_str}, decr_count_over_{second_str}, decr_count_over_{third_str}) 
                    SELECT {groupby}, min(diff_mean) AS min_diff_mean, avg(diff_mean) AS mean_diff_mean, 
                    max(diff_mean) AS max_diff_mean, 
                    sum(CASE WHEN diff_mean >= {0}*{multiplier} THEN 1 ELSE 0 END) AS incr_count_over_{first_str},
                    sum(CASE WHEN diff_mean >= {1}*{multiplier} THEN 1 ELSE 0 END) AS incr_count_over_{second_str},
                    sum(CASE WHEN diff_mean >= {2}*{multiplier} THEN 1 ELSE 0 END) AS incr_count_over_{third_str},
                    sum(CASE WHEN diff_mean <= -{0}*{multiplier} THEN 1 ELSE 0 END) AS decr_count_over_{first_str},
                    sum(CASE WHEN diff_mean <= -{1}*{multiplier} THEN 1 ELSE 0 END) AS decr_count_over_{second_str},
                    sum(CASE WHEN diff_mean <= -{2}*{multiplier} THEN 1 ELSE 0 END) AS decr_count_over_{third_str}
                    FROM 
                    (SELECT t1.from_stop_I AS from_stop_I, t1.to_stop_I AS to_stop_I, t2.mean-t1.mean AS diff_mean 
                    FROM before.{measure} AS t1, after.{measure} AS t2 
                    WHERE t1.from_stop_I = t2.from_stop_I AND t1.to_stop_I = t2.to_stop_I {ignore_stops} 
                    AND abs(t2.mean-t1.mean) < {threshold}) q1 
                    GROUP BY {groupby}""".format(first, second, third,
                                                 first_str=first_str, second_str=second_str, third_str=third_str,
                                                 measure=measure,
                                                 groupby=groupby, multiplier=multiplier, threshold=threshold,
                                                 ignore_stops=ignore_stops)

        cur.execute(query)
        self.conn.commit()

    def get_mean_change_for_all_targets(self, groupby="to_stop_I", measure="temporal_distance", ignore_stops=None):
        """
        Returns pre generated differences table as pandas DataFrame
        :param groupby: "to_stop_I" or "from_stop_I" designating if calculating the measure to the target or from the target
        :param measure: "temporal_distance", "n_boardings",
        :return:

        if ignore_stops:
            ignore_stops = " WHERE " + groupby + " IN " + ignore_stops
        else:
            ignore_stops = ""
        """
        query = """SELECT * FROM diff_{groupby}_{measure}""".format(measure=measure, groupby=groupby)
        print("running query")
        df = pandas.read_sql_query(query, self.conn)
        df = self.gtfs.add_coordinates_to_df(df, stop_id_column=groupby, lat_name="lat", lon_name="lon")
        if measure == "temporal_distance":
            df["mean_diff_mean"] = df["mean_diff_mean"].apply(lambda x: x / 60)

        return df

    def extreme_change_od_pairs(self, threshold):
        """
        Returns O-D pairs where the absolute change is larger than the threshold. Returns increase in travel time with
        positive thresholds and decrease in travel time with negative thresholds
        :param threshold: int
        :return: Pandas DataFrame
        """
        if threshold < 0:
            string_to_add = " <= " + str(threshold)
        else:
            string_to_add = " >= " + str(threshold)

        query = """SELECT t1.from_stop_I AS from_stop_I, t1.to_stop_I AS to_stop_I, t2.mean-t1.mean AS diff_mean 
                    FROM before.temporal_distance AS t1, after.temporal_distance AS t2 
                    WHERE t1.from_stop_I = t2.from_stop_I AND t1.to_stop_I = t2.to_stop_I 
                    AND t2.mean-t1.mean %s AND t2.mean-t1.mean < 10800""" % (string_to_add,)
        df = pandas.read_sql_query(query, self.conn)
        return df

    def get_global_mean_change(self, measure, threshold=10800, ignore_stops=False):
        ignore_list = ""

        if ignore_stops:
            ignore_list=stops_to_exclude(return_sqlite_list=True)
        query = """SELECT before_global_mean, after_global_mean, after_global_mean-before_global_mean AS global_mean_difference FROM
                   (SELECT avg(mean) AS before_global_mean FROM before.{measure} WHERE mean <= {threshold} AND mean >0 
                   AND from_stop_I NOT IN {ignore_stops} AND to_stop_I NOT IN {ignore_stops}) t1,
                   (SELECT avg(mean) AS after_global_mean FROM after.{measure} WHERE mean <= {threshold} AND mean >0 
                   AND from_stop_I NOT IN {ignore_stops} AND to_stop_I NOT IN {ignore_stops}) t2
                   """.format(measure=measure, threshold=threshold, ignore_stops=ignore_list)
        df = pandas.read_sql_query(query, self.conn)
        return df

    @timeit
    def get_rows_with_abs_change_greater_than_n(self, stops, measure, n, sign, unit="s"):
        stops = ",".join([str(x) for x in stops])
        divisors = {"s": 1, "m": 60, "h": 3600}
        divisor = divisors[unit]
        query = """SELECT t1.{measure}/{divisor} AS before_{measure}, t2.{measure}/{divisor} AS after_{measure}, 
                    (t2.{measure}-t1.{measure})/{divisor} AS diff_{measure} FROM before.temporal_distance AS t1, 
                    after.temporal_distance AS t2 
                    WHERE t1.from_stop_I != t1.to_stop_I AND t1.from_stop_I = t2.from_stop_I 
                    AND t1.to_stop_I = t2.to_stop_I AND t1.from_stop_I NOT IN ({stops}) 
                    AND t2.to_stop_I NOT IN ({stops}) 
                    AND t2.{measure}-t1.{measure} {sign} {n}""".format(measure=measure,
                                                                       divisor=divisor,
                                                                       stops=stops,
                                                                       n=n,
                                                                       sign=sign)

        df = pandas.read_sql_query(query, self.conn)

        return df

    @timeit
    def get_rows_based_on_stop_list(self, from_stops, to_stops, measure, measure_mode, unit="s"):
        """

        :param from_stops: list
        :param to_stops: list
        :param measure: string (mean, min, max, median)
        :param unit: string
        :param measure_mode: string
        :return:
        """
        assert measure_mode in ["n_boardings", "temporal_distance"]
        from_stops = ",".join([str(x) for x in from_stops])
        to_stops = ",".join([str(x) for x in to_stops])
        divisors = {"s": 1, "m": 60, "h": 3600}
        divisor = divisors[unit]

        query = """SELECT t1.{measure}/{divisor} AS before_{measure}, t2.{measure}/{divisor} AS after_{measure}, 
                    (t2.{measure}-t1.{measure})/{divisor} AS diff_{measure} FROM before.{mode} AS t1, 
                    after.{mode} AS t2 
                    WHERE t1.from_stop_I != t1.to_stop_I AND t1.from_stop_I = t2.from_stop_I 
                    AND t1.to_stop_I = t2.to_stop_I AND t1.from_stop_I IN ({from_stops}) 
                    AND t2.to_stop_I IN ({to_stops})""".format(measure=measure,
                                                               mode=measure_mode,
                                                               divisor=divisor,
                                                               from_stops=from_stops,
                                                               to_stops=to_stops)

        df = pandas.read_sql_query(query, self.conn)

        return df

    def get_data_for_target(self, target, measure, direction="to", threshold=10800, unit="s", ignore_stops=False):
        divisors = {"s": 1, "m": 60, "h": 3600}
        divisor = divisors[unit]
        ignore_list = ""
        if ignore_stops:
            ignore_list = stops_to_exclude(return_sqlite_list=True)
            ignore_list = " AND t1.from_stop_I NOT IN {ignore_list} AND t1.to_stop_I NOT IN {ignore_list}".format(ignore_list=ignore_list)

        query = """SELECT t1.from_stop_I, t1.to_stop_I, t1.mean/{divisor} AS before_mean, t2.mean/{divisor} AS after_mean, 
                    (t2.mean-t1.mean)/{divisor} AS diff_mean, COALESCE((t2.mean/t1.mean)- 1, 0) AS diff_mean_relative
                    FROM before.{measure} t1, after.{measure} t2
                    WHERE t1.from_stop_I=t2.from_stop_I AND t1.to_stop_I=t2.to_stop_I AND t1.mean <= {threshold} 
                    AND t2.mean <= {threshold} 
                    AND t1.{direction}_stop_I={target} {ignore_list}""".format(measure=measure,
                                                                               target=target,
                                                                               direction=direction,
                                                                               threshold=threshold,
                                                                               divisor=divisor,
                                                                               ignore_list=ignore_list)

        df = pandas.read_sql_query(query, self.conn)
        return df

    def get_mean_change(self, measure, threshold=10800, descening_order=False, include_list=None):

        if descening_order:
            order_by = "DESC"
        else:
            order_by = "ASC"
        include_list = "(" + ",".join([str(x) for x in include_list]) + ")"
        query = """SELECT t1.to_stop_I, t2.mean AS before, t2.mean-t1.mean AS diff_mean FROM 
                    (SELECT to_stop_I, avg(mean) AS mean FROM before.{measure}
                     WHERE mean <= {threshold} AND to_stop_I IN {include_list}
                    GROUP BY to_stop_I) t1, 
                    (SELECT to_stop_I, avg(mean) AS mean FROM after.{measure}
                     WHERE mean <= {threshold}  AND to_stop_I IN {include_list}
                    GROUP BY to_stop_I) t2
                    WHERE t1.to_stop_I=t2.to_stop_I
                    ORDER BY diff_mean {order_by}
                    """.format(measure=measure,
                               threshold=threshold,
                               order_by=order_by,
                               include_list=include_list)

        df = pandas.read_sql_query(query, self.conn)
        return df

    def get_n_winning_targets_using_change_in_mean(self, n, measure, distance=500, threshold=10800, losers=False, include_list=None):

        if losers:
            order_by = "DESC"
        else:
            order_by = "ASC"
        include_list = "(" + ",".join([str(x) for x in include_list]) + ")"
        query = """SELECT t1.to_stop_I, t2.mean-t1.mean AS diff_mean FROM 
                    (SELECT to_stop_I, avg(mean) AS mean FROM before.{measure}
                     WHERE mean <= {threshold} AND to_stop_I IN {include_list}
                    GROUP BY to_stop_I) t1, 
                    (SELECT to_stop_I, avg(mean) AS mean FROM after.{measure}
                     WHERE mean <= {threshold}  AND to_stop_I IN {include_list}
                    GROUP BY to_stop_I) t2
                    WHERE t1.to_stop_I=t2.to_stop_I
                    ORDER BY diff_mean {order_by}
                    """.format(measure=measure,
                               threshold=threshold,
                               order_by=order_by,
                               include_list=include_list)

        df = pandas.read_sql_query(query, self.conn)
        # exclude nearby stops
        nearby_excluded_stops = []
        stops_remaining = []
        gtfs = GTFS(GTFS_PATH)
        for value in df.itertuples():
            if not value.to_stop_I in nearby_excluded_stops:
                exclude_df = gtfs.get_stops_within_distance(value.to_stop_I, distance)
                nearby_excluded_stops += list(exclude_df["stop_I"])
                stops_remaining.append(value.to_stop_I)
                if len(stops_remaining) == n:
                    break
        df = df.loc[df['to_stop_I'].isin(stops_remaining)]
        return df

    def n_inf_stops_per_stop(self, measure, indicator, threshold, group_by="to_stop_I", routing="before"):
        if group_by == "to_stop_I":
            stop_I = "from_stop_I"
        elif group_by == "from_stop_I":
            stop_I = "to_stop_I"
        else:
            raise AssertionError("Group_by should be to_stop_I or from_stop_I")

        query = """SELECT {group_by}, count(to_stop_I) AS N_stops FROM {routing}.{measure}
                    WHERE {indicator} >{threshold}  
                    GROUP by {group_by} ORDER BY count(to_stop_I)""".format(measure=measure,
                                                                            threshold=threshold,
                                                                            indicator=indicator,
                                                                            routing=routing,
                                                                            group_by=group_by,
                                                                            stop_I=stop_I)
        df = pandas.read_sql_query(query, self.conn)
        return df

    def find_stops_where_all_indicators_are_finite(self, measure="temporal_distance", indicator="max", routing="after",
                                                   threshold=10800):
        stops_to_ignore = []
        ignore_statement = ""
        while True:
            query = """SELECT from_stop_I, count(to_stop_I) as invalid_connections FROM {routing}.{measure} 
                       WHERE {indicator} >= {threshold} {ignore_statement} group by from_stop_I order by invalid_connections""".format(measure=measure,
                                                                                                        indicator=indicator,
                                                                                                        threshold=threshold,
                                                                                                        routing=routing,
                                                                                                        ignore_statement=ignore_statement)
            df = pandas.read_sql_query(query, self.conn)
            print("query has run, with {n} stops remaining".format(n=len(df.index)))
            df['removal_column'] = df.index+df.invalid_connections
            n_stops_in_iteration = len(df.index)
            df_to_remove = df.loc[df['removal_column'] > n_stops_in_iteration]
            print("{n} stops removed".format(n=len(df_to_remove.index)))

            if len(df_to_remove.index) == 0:
                break
            stops_to_ignore += list(df_to_remove['from_stop_I'])
            stops_to_ignore_str = ""
            for stop in stops_to_ignore:
                if not stops_to_ignore_str == "":
                    stops_to_ignore_str += ","
                stops_to_ignore_str += str(stop)
            # stops_to_ignore_str = ','.join(stops_to_ignore_str)
            ignore_statement = "AND from_stop_I NOT IN ({stops_comma}) " \
                               "AND to_stop_I NOT IN ({stops_comma})".format(stops_comma=stops_to_ignore_str)
        return list(df['from_stop_I']), stops_to_ignore

    def find_stops_where_all_indicators_are_finite_using_network(self, measure="temporal_distance", indicator="max",
                                                       routing="after",
                                                       threshold=10800):
        pass
        """
        nodes = [x[0] for x in nodes]
        edges = itertools.combinations(nodes, 2)
        print("combinations")

        G = nx.Graph()
        G.add_edges_from(edges)
        print("initial edges in place")

        for row in df.iterrows():
            G.remove_edge(row.from_stop_I, row.to_stop_I)
            print("removing stuff")
        """

if __name__ == "__main__":
    for time in TIMES:
        a2aa = AllToAllDifferenceAnalyzer(GTFS_PATH, get_a2aa_db_path(time, "old"), get_a2aa_db_path(time, "lm"),
                                          get_a2aa_db_path(time, "output"))
        ignore_list = stops_to_exclude(return_sqlite_list=True)
        a2aa.diff_table(groupby="to_stop_I", measure="n_boardings", ignore_stops=ignore_list)
        a2aa.diff_table(groupby="from_stop_I", measure="n_boardings", ignore_stops=ignore_list)
        a2aa.diff_table(groupby="to_stop_I", measure="temporal_distance", ignore_stops=ignore_list)
        a2aa.diff_table(groupby="from_stop_I", measure="temporal_distance", ignore_stops=ignore_list)
        #a2aa.diff_table(groupby="to_stop_I", measure="journey_duration", ignore_stops=ignore_list)
        #a2aa.diff_table(groupby="from_stop_I", measure="journey_duration", ignore_stops=ignore_list)