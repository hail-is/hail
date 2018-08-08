

# represents a rectangular region
class Zone(object):

    def __init__(self, a_range, b_range):
        if not (len(a_range) == len(b_range) == 2):
            raise ValueError(
                "Failed to initialize Zone: ranges must have length 2.")
        if not a_range[0] <= a_range[1]:
            raise ValueError("Failed to initialize Zone: a_range not valid.")
        if not b_range[0] <= b_range[1]:
            raise ValueError("Failed to initialize Zone: b_range not valid.")
        self.a_range = a_range
        self.b_range = b_range

    @staticmethod
    def _within(_range, container):
        if _range[0] >= container[0] and _range[1] <= container[1]:
            return True
        return False

    def within(self, zone):
        return (self._within(self.a_range, zone.a_range)
                and self._within(self.b_range, zone.b_range))

    def contains(self, zone):
        return (zone._within(zone.a_range, self.a_range)
                and zone._within(zone.b_range, self.b_range))


class Zones:

    def __init__(self, *zones):
        self.zones = []
        self.zones.extend(zones)

    def contains(self, zone):
        return any(z.contains(zone) for z in self.zones)

    def append(self, zone):
        self.zones.append(zone)