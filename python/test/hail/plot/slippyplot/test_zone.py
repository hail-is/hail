import unittest

from hail.plot.slippyplot.zone import Zone, Zones

from test.hail.helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class Tests(unittest.TestCase):
    def setUp(self):
        self.a = Zone([0, 1], [5, 20])
        self.b = Zone([-5, 10], [-1, -1])
        self.zs = Zones(*[self.a, self.b])

    def test_good_zone_initialization(self):
        z = Zone([0, 3], [4, 10])
        self.assertEqual(z.a_range, [0, 3])
        self.assertEqual(z.b_range, [4, 10])

    def test_bad_zone_initialization(self):
        self.assertRaises(ValueError, Zone, [3, 0], [4, 10])
        self.assertRaises(ValueError, Zone, [0, 3], [10, 4])
        self.assertRaises(ValueError, Zone, [0, 3], [-10, -11])

    def test_within_helper(self):
        z = Zone([0, 0], [0, 0])
        self.assertTrue(z._within([1, 10], [1, 10]))
        self.assertTrue(z._within([2, 10], [1, 10]))
        self.assertFalse(z._within([0, 10], [1, 10]))

        self.assertTrue(z._within([1, 9], [1, 10]))
        self.assertTrue(z._within([1, 10], [1, 10]))
        self.assertFalse(z._within([1, 11], [1, 10]))

    def test_within(self):
        z = Zone([2, 5], [-11, 3])
        x = Zone([2, 5], [-10, 1])
        self.assertTrue(x.within(z))
        self.assertFalse(z.within(x))

    def test_contains(self):
        z = Zone([2, 5], [-11, 3])
        x = Zone([2, 5], [-10, 1])
        self.assertTrue(z.contains(x))
        self.assertFalse(x.contains(z))

    def test_zones_initialization(self):
        self.assertEqual(self.zs.zones[0], self.a)
        self.assertEqual(self.zs.zones[1], self.b)

    def test_zones_contains(self):
        x1 = Zone([0, 1], [6, 17])
        x2 = Zone([-1, 3], [18, 19])
        x3 = Zone([-3, 0], [-2, -1])
        self.assertTrue(self.zs.contains(self.a))
        self.assertTrue(self.zs.contains(x1))
        self.assertFalse(self.zs.contains(x2))
        self.assertFalse(self.zs.contains(x3))

    def test_zones_append(self):
        z = Zone([0, 3], [2, 9])
        self.zs.append(z)
        self.assertTrue(self.zs.contains(z))
