import unittest
import os
from shapely.wkt import loads
from gtfspy.gtfs import GTFS
from research.westmetro_paper.scripts.routemap_cluster_the_shapes import RouteMapMaker


class TestRoutemap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ This method is run once before executing any tests"""
        cls.gtfs_source_dir = "../../../../gtfspy/gtfspy/test/test_data"
        cls.G = GTFS.from_directory_as_inmemory_db(cls.gtfs_source_dir)

    def setUp(self):
        """This method is run once before _each_ test method is executed"""
        self.gtfs = GTFS.from_directory_as_inmemory_db(self.gtfs_source_dir)
        self.RMM = RouteMapMaker(self.gtfs)

    def test_split_shape_by_points(self):
        shape = loads('''LINESTRING(367521.7699326264
        6672719.639501401, 367541.1158726254
        6672704.783398243, 367576.7212629453
        6672674.17991306, 367599.8762514296
        6672654.058925985, 367605.4661413688
        6672645.831208155, 367609.0422289064
        6672635.669883275, 367609.1306812162
        6672616.605782185, 367618.9299698949
        6672611.124081238, 367630.329573809
        6672597.670395718, 367666.1890881368
        6672527.933707527, 367677.4934538465
        6672513.368960739, 367689.2020455575
        6672483.853055792, 367721.3621130693
        6672465.414051151)''')
        parent_stops = [3604, 3609]
        point_data = ['POINT (367724.7651578544 6672464.287952477)',
                       'POINT (367609.0422289064 6672635.669883275)',
                       'POINT (367504.0169496978 6672737.837889094)']
        points = [loads(x) for x in point_data]
        point_ids = (890.0, 123, 513.0)
        row = {"geometry": shape, "from_stop_I": parent_stops[0], "to_stop_I": parent_stops[1], "point_geom": points,
               "new_stop_I": point_ids}
        row = self.RMM.split_shape_by_points(row)

        shape_parts, stop_sections = row["shape_parts"], row["child_stop_Is"]
        self.assertEqual(len(shape_parts), len(stop_sections))
        print(shape_parts, stop_sections)
        self.assertFalse(all(x == shape_parts[0] for x in shape_parts))



if __name__ == '__main__':
    unittest.main()
