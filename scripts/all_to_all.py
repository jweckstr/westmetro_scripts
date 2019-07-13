import os
import sys
import pickle
import gc
from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler
from gtfspy.gtfs import GTFS
from gtfspy.util import timeit
from gtfspy.networks import walk_transfer_stop_to_stop_network
from gtfspy.util import makedirs
from gtfspy.routing.connection import Connection
from gtfspy.routing.journey_data import JourneyDataManager, DiffDataManager
from research.westmetro_paper.scripts.util import split_into_equal_length_parts
from research.westmetro_paper.scripts.all_to_all_settings import *

"""
Loops trough a set of target nodes, runs routing for them and stores results in a database
Pipeline
pyfile1
1. Create database, based on the parameters set in settings
2. Divide origin nodes into n parts
3. Run all_to_all.py
4a. pickle
or
4b. direct to db
5. Create indicies once everything is finished
all_to_all.py

srun --mem=1G --time=0:10:00 python3 research/westmetro_paper/scripts/all_to_all.py run_preparations

srun --mem=6G --time=2:00:00 python3 research/westmetro_paper/scripts/all_to_all.py to_db
"""


class AllToAllRoutingPipeline:
    def __init__(self, feed_dict, routing_params):
        self.pickle = PICKLE
        self.gtfs_dir = feed_dict["gtfs_dir"]
        self.G = GTFS(feed_dict["gtfs_dir"])
        self.tz = self.G.get_timezone_name()
        self.journey_dir = feed_dict["journey_dir"]
        self.day_start = feed_dict["day_start"]
        self.day_end = feed_dict["day_end"]
        self.routing_start_time = feed_dict["routing_start_time"]
        self.routing_end_time = feed_dict["routing_end_time"]
        self.analysis_start_time = feed_dict["analysis_start_time"]
        self.analysis_end_time = feed_dict["analysis_end_time"]
        self.pickle_dir = feed_dict["pickle_dir"]
        self.routing_params = routing_params

        self.jdm = None
        if not self.pickle:
            self.jdm = JourneyDataManager(os.path.join(GTFS_DB_WORK_DIR, GTFS_DB_FNAME),
                                          journey_db_path=os.path.join(RESULTS_DIR, JOURNEY_DB_FNAME),
                                          routing_params=self.routing_params, track_vehicle_legs=TRACK_VEHICLE_LEGS,
                                          track_route=TRACK_ROUTE)

    def get_all_events(self):
        print("Retrieving transit events")
        connections = []
        for e in self.G.generate_routable_transit_events(start_time_ut=self.routing_start_time,
                                                         end_time_ut=self.routing_end_time):
            connections.append(Connection(int(e.from_stop_I),
                                          int(e.to_stop_I),
                                          int(e.dep_time_ut),
                                          int(e.arr_time_ut),
                                          int(e.trip_I),
                                          int(e.seq)))
        assert (len(connections) == len(set(connections)))
        print("scheduled events:", len(connections))
        print("Retrieving walking network")
        net = walk_transfer_stop_to_stop_network(self.G, max_link_distance=CUTOFF_DISTANCE)
        print("net edges: ", len(net.edges()))
        return net, connections

    @timeit
    def loop_trough_targets_and_run_routing(self, targets, slurm_array_i):
        net, connections = self.get_all_events()
        csp = None

        for target in targets:
            print(target)
            if csp is None:
                csp = MultiObjectivePseudoCSAProfiler(connections, target, walk_network=net,
                                                      end_time_ut=self.routing_end_time,
                                                      transfer_margin=TRANSFER_MARGIN,
                                                      start_time_ut=self.routing_start_time, walk_speed=WALK_SPEED,
                                                      verbose=True, track_vehicle_legs=TRACK_VEHICLE_LEGS,
                                                      track_time=TRACK_TIME, track_route=TRACK_ROUTE)
            else:
                csp.reset([target])
            csp.run()

            profiles = dict(csp.stop_profiles)
            if self.pickle:
                self._pickle_results(profiles, slurm_array_i, target)
            else:
                self.jdm.import_journey_data_for_target_stop(target, profiles)
            profiles = None
            gc.collect()

    @timeit
    def loop_trough_targets_and_run_routing_with_route(self, targets, slurm_array_i):
        net, connections = self.get_all_events()
        csp = None

        for target in targets:
            print("target: ", target)
            if csp is None:
                csp = MultiObjectivePseudoCSAProfiler(connections, target, walk_network=net,
                                                      end_time_ut=self.routing_end_time,
                                                      transfer_margin=TRANSFER_MARGIN,
                                                      start_time_ut=self.routing_start_time, walk_speed=WALK_SPEED,
                                                      verbose=True, track_vehicle_legs=TRACK_VEHICLE_LEGS,
                                                      track_time=TRACK_TIME, track_route=TRACK_ROUTE)
            else:
                csp.reset([target])
            csp.run()

            profiles = dict(csp.stop_profiles)
            if self.pickle:
                self._pickle_results(profiles, slurm_array_i, target)
            else:
                self.jdm.import_journey_data_for_target_stop(target, profiles)
            profiles = None
            gc.collect()
    @timeit
    def _pickle_results(self, profiles, pickle_subdir, target):
        pickle_path = makedirs(os.path.join(self.pickle_dir, str(pickle_subdir)))
        pickle_path = os.path.join(pickle_path, str(target) + ".pickle")
        profiles = dict((key, value.get_final_optimal_labels()) for (key, value) in profiles.items())
        """for key, values in profiles.items():
            values.sort(key=lambda x: x.departure_time, reverse=True)
            new_values = compute_pareto_front(values)
            profiles[key] = new_values
            """
        pickle.dump(profiles, open(pickle_path, 'wb'), -1)
        profiles = None
        gc.collect()

    def get_list_of_stops(self, where=''):
        df = self.G.execute_custom_query_pandas("SELECT stop_I FROM stops " + where + " ORDER BY stop_I")
        return df

    @timeit
    def store_pickle_in_db(self):
        self.jdm = JourneyDataManager(self.gtfs_dir, journey_db_path=self.journey_dir,
                                      routing_params=self.routing_params, track_vehicle_legs=TRACK_VEHICLE_LEGS,
                                      track_route=TRACK_ROUTE)
        for root, dirs, files in os.walk(self.pickle_dir):
            for target_file in files:
                target = target_file.replace(".pickle", "")
                if not target in self.jdm.get_targets_having_journeys():
                    print("target: ", target)
                    profiles = pickle.load(open(os.path.join(root, target_file), 'rb'))

                    self.jdm.import_journey_data_for_target_stop(int(target), profiles)
                else:
                    print("skipping: ", target, " already in db")

        self.jdm.create_indices()

    def calculate_additional_columns_for_journey(self):
        if not self.jdm:
            self.jdm = JourneyDataManager(self.gtfs_dir, journey_db_path=self.journey_dir,
                                          routing_params=self.routing_params, track_vehicle_legs=TRACK_VEHICLE_LEGS,
                                          track_route=TRACK_ROUTE)
        self.jdm.populate_additional_journey_columns()
        self.jdm.compute_and_store_travel_impedance_measures(self.analysis_start_time, self.analysis_end_time, TRAVEL_IMPEDANCE_STORE_PATH)

    def calculate_comparison_measures(self):
        if not self.jdm:
            self.jdm = JourneyDataManager(self.gtfs_dir, journey_db_path=self.journey_dir,
                                          routing_params=self.routing_params, track_vehicle_legs=TRACK_VEHICLE_LEGS,
                                          track_route=TRACK_ROUTE)
        prev_dict = None
        prev_key = None
        before_db_tuple = None
        after_db_tuple = None
        for (key, feed_dict) in FEED_LIST:
            if prev_dict:
                if feed_dict["feed_seq"] < prev_dict["feed_seq"]:
                    after_db_tuple = (feed_dict["journey_dir"], key)
                    before_db_tuple = (prev_dict["journey_dir"], prev_key)
                else:
                    before_db_tuple = (feed_dict["journey_dir"], key)
                    after_db_tuple = (prev_dict["journey_dir"], prev_key)
            prev_dict = feed_dict
            prev_key = key

        self.jdm.initialize_comparison_tables(DIFF_PATH, before_db_tuple, after_db_tuple)


def main(cmd, args):
    print(cmd)
    if cmd == "run_preparations":
        for (feed, feed_dict) in FEED_LIST:
            jdm = JourneyDataManager(feed_dict["gtfs_dir"], None, routing_params(feed),
                                     track_vehicle_legs=TRACK_VEHICLE_LEGS, track_route=TRACK_ROUTE)
            jdm.initialize_database(feed_dict["journey_dir"])

    elif cmd[:5] == "run_w":
        feed = args[0]
        slurm_array_i = args[1]
        slurm_array_length = args[2]
        slurm_array_i = int(slurm_array_i)
        slurm_array_length = int(slurm_array_length)

        assert(slurm_array_i < slurm_array_length)

        if cmd == "run_routing":
            a2a_pipeline = AllToAllRoutingPipeline()
            df = a2a_pipeline.get_list_of_stops()
            nodes = df['stop_I'].sample(frac=1, random_state=123).values
            parts = split_into_equal_length_parts(nodes, slurm_array_length)
            targets = parts[slurm_array_i]
            assert not TRACK_ROUTE
            a2a_pipeline.loop_trough_targets_and_run_routing(targets, slurm_array_i)
        elif cmd == "run_with_routes_slurm":
            print(feed)
            a2a_pipeline = AllToAllRoutingPipeline(FEED_DICT[feed], routing_params(feed))
            df = a2a_pipeline.get_list_of_stops(where="WHERE code = '__target__'")
            nodes = df['stop_I'].sample(frac=1, random_state=123).values
            parts = split_into_equal_length_parts(nodes, slurm_array_length)
            targets = parts[slurm_array_i]
            assert TRACK_ROUTE
            a2a_pipeline.loop_trough_targets_and_run_routing_with_route(targets, slurm_array_i)

    elif cmd == "run_everything":
        assert TRACK_ROUTE
        targets = [1040]
        for (feed, feed_dict) in FEED_LIST:
            #if feed == "old_daily":
            #    continue
            print("feed: ", feed)
            a2a_pipeline = AllToAllRoutingPipeline(feed_dict, routing_params(feed))
            if True:
                a2a_pipeline.loop_trough_targets_and_run_routing_with_route(targets, "testing")
                a2a_pipeline.store_pickle_in_db()
            a2a_pipeline.calculate_additional_columns_for_journey()

    elif cmd == "diff_db":
        tables = ["n_boardings",
                  "journey_duration",
                  "in_vehicle_duration",
                  "transfer_wait_duration",
                  "walking_duration",
                  "pre_journey_wait_fp",
                  "temporal_distance"]
        diff = DiffDataManager(DIFF_PATH)
        diff.initialize_journey_comparison_tables(tables, (FEED_DICT["old_daily"]["journey_dir"], "old_daily"),
                                                  (FEED_DICT["lm_daily"]["journey_dir"], "lm_daily"))
    elif cmd == "to_db":
        a2a_pipeline = AllToAllRoutingPipeline()
        a2a_pipeline.store_pickle_in_db()

    elif cmd == "post_import":
        a2a_pipeline = AllToAllRoutingPipeline()
        a2a_pipeline.calculate_additional_columns_for_journey()

    elif cmd == "od_measures":
        a2a_pipeline = AllToAllRoutingPipeline()
        a2a_pipeline.calculate_comparison_measures()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

