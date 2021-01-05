from ..expr.types import tstruct


class PlacementTree:
    def __init__(self, name, width, height, children):
        self.name = name
        self.width = width
        self.height = height
        self.children = children

    def __repr__(self):
        return f'PlacementTree({self.name}, {self.width}, {self.height}, {self.children})'

    @staticmethod
    def from_named_type(name, dtype):
        if not isinstance(dtype, tstruct):
            return PlacementTree(name, 1, 0, [])
        children = [PlacementTree.from_named_type(name, dtype) for name, dtype in dtype.items()]
        width = sum(child.width for child in children)
        height = max([child.height for child in children], default=0) + 1
        return PlacementTree(name, width, height, children)

    def to_grid(self):
        grid = []
        current_height = self.height
        frontier = self.children
        while any(x.height != current_height for x in frontier):
            new_frontier = []
            row = []
            grid.append(row)
            for x in frontier:
                if x.height == current_height:
                    row.append((x.name, x.width))
                    new_frontier += x.children
                else:
                    row.append((None, x.width))
                    new_frontier.append(x)
                    frontier = new_frontier
            current_height -= 1
        row = []
        grid.append(row)
        for x in frontier:
            assert x.height == current_height, (x.height, current_height)
            assert x.width == 1, x.width
            row.append((x.name, x.width))
        return grid
