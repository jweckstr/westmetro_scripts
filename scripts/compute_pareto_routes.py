import pickle

from pandas import DataFrame

from gtfspy.routing.models import Connection
from gtfspy.gtfs import GTFS
from gtfspy.routing.journey_data import JourneyDataManager
from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler
from gtfspy.routing.node_profile_analyzer_time_and_veh_legs import NodeProfileAnalyzerTimeAndVehLegs
from gtfspy.util import makedirs
from scripts.generic_settings import *
"""
Generic travel time matrix pipeline:
Importation of gtfs database
Generate the required events, nodes and stop distances
Do routing
Store in suitable format (sqlite etc)

"""


def target_list_to_str(targets):
    targets_str = "_".join([str(target) for target in targets])
    return targets_str


class GenericJourneyDataPipeline:
    def __init__(self):
        self.G = GTFS(GTFS_DATA_BASEDIR)
        self.day_start_ut = self.G.get_suitable_date_for_daily_extract(ut=True)+3600
        self.start_time = self.day_start_ut+8*3600
        self.end_time = self.day_start_ut+11*3600
        self.profiles = {}
        self.journey_analyzer = None
        # self.analysis_start_time
        # self.analysis_end_time
        makedirs(RESULTS_DIRECTORY)
        print("Retrieving transit events")
        self.connections = []
        for e in self.G.generate_routable_transit_events(start_time_ut=self.start_time, end_time_ut=self.end_time):
            self.connections.append(Connection(int(e.from_stop_I),
                                               int(e.to_stop_I),
                                               int(e.dep_time_ut),
                                               int(e.arr_time_ut),
                                               int(e.trip_I)))
        print("Retrieving walking network")
        self.net = self.G.get_walk_transfer_stop_to_stop_network()

    def script(self):

        self.get_profile_data()
        journey_analyzer = JourneyDataManager(TARGET_STOPS, JOURNEY_DATA_DIR, GTFS_DATA_BASEDIR, ROUTING_PARAMS,
                                              track_route=True, close_connection=False)
        journey_analyzer.import_journey_data_for_target_stop(self.profiles)
        journey_analyzer.create_indices()
        if False:
            journey_analyzer.add_fastest_path_column()

        """
        all_geoms = journey_analyzer.get_all_geoms()
        journey_path = os.path.join(RESULTS_DIRECTORY, "all_routes_to_" + target_list_to_str(TARGET_STOPS) + ".geojson")
        with open(journey_path, 'w') as f:
            dump(journey_analyzer.extract_geojson(all_geoms), f)
        """

    def get_profile_data(self, targets=TARGET_STOPS, recompute=False):
        node_profiles_fname = os.path.join(RESULTS_DIRECTORY, "node_profile_" + target_list_to_str(targets) + ".pickle")
        if not recompute and os.path.exists(node_profiles_fname):
            print("Loading precomputed data")
            self.profiles = pickle.load(open(node_profiles_fname, 'rb'))
            print("Loaded precomputed data")
        else:
            print("Recomputing profiles")
            self._compute_profile_data()
            pickle.dump(self.profiles, open(node_profiles_fname, 'wb'), -1)
            print("Recomputing profiles")

    def _compute_profile_data(self):
        csp = MultiObjectivePseudoCSAProfiler(self.connections, TARGET_STOPS, walk_network=self.net,
                                              transfer_margin=TRANSFER_MARGIN, walk_speed=WALK_SPEED, verbose=True,
                                              track_vehicle_legs=False, track_time=True, track_route=True)
        print("CSA Profiler running...")
        csp.run()
        print("CSA profiler finished")

        self.profiles = dict(csp.stop_profiles)

    def key_measures_as_csv(self, csv_path="stop_data.csv"):
        """
        Combines key temporal distance measures for each node with stop data from gtfs and stores in csv format
        :return:
        """
        node_profiles_list = []
        # iterate through all node profiles and add the NodeProfileAnalyzer data to a list of dicts
        for node, profile in self.profiles.items():
            npa = NodeProfileAnalyzerTimeAndVehLegs.from_profile(profile, self.start_time, self.end_time)
            node_profile_dict = npa.get_node_profile_measures_as_dict()
            node_profile_dict["node"] = node
            node_profiles_list.append(node_profile_dict)

        node_profiles = DataFrame(node_profiles_list)
        stops = self.G.stops()
        stops.join(node_profiles.set_index("node"), on='stop_I').to_csv(path_or_buf=csv_path)


def main():
    gap = GenericJourneyDataPipeline()
    gap.script()

if __name__ == "__main__":
    main()