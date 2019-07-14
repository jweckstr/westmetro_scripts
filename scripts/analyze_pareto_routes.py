from gtfspy.gtfs import GTFS

from gtfspy.routing.journey_data_analyzer import JourneyDataAnalyzer
from scripts.generic_settings import *



class GenericJourneyDataAnalysisPipeline:
    def __init__(self):
        self.G = GTFS(GTFS_DATA_BASEDIR)
        self.day_start_ut = self.G.get_suitable_date_for_daily_extract(ut=True) + 3600
        self.start_time = self.day_start_ut + 8 * 3600
        self.end_time = self.day_start_ut + 11 * 3600
        self.profiles = {}
        self.journey_analyzer = None
        # self.analysis_start_time
        # self.analysis_end_time


    def script(self):

        journey_analyzer = JourneyDataAnalyzer(JOURNEY_DATA_DIR, GTFS_DATA_BASEDIR)
        if False:
            gdf = journey_analyzer.get_transfer_stops()
            gdf.to_file(shapefile_dir('transfer_stops'), driver='ESRI Shapefile')
            gdf = journey_analyzer.get_transfer_walks()
            gdf.to_file(shapefile_dir('transfer_walks'), driver='ESRI Shapefile')
            gdf = journey_analyzer.journeys_per_section()
            gdf.to_file(shapefile_dir('journeys_per_section'), driver='ESRI Shapefile')
            gdf = journey_analyzer.journey_alternatives_per_stop()
            gdf.to_file(shapefile_dir('journeys_per_stop'), driver='ESRI Shapefile')
        journey_analyzer.n_route_alternatives()


def main():
    gap = GenericJourneyDataAnalysisPipeline()
    gap.script()


if __name__ == "__main__":
    main()