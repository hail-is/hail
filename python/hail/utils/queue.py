from collections import Sequence


class Queue(Sequence):
    def __init__(self):
        self._list = []
        self._len = 0
        self._consumed = False

    def push(self, *items):
        q = Queue()
        if self._consumed:
            # FIXME: full copy on append is still quadratic, but now it can be fixed in one place
            q._list = self._list[:self._len]
        else:
            # share list for the first copy
            self._consumed = True
            q._list = self._list
        q._len = self._len
        q._consumed = False
        for x in items:
            q._list.append(x)
            q._len += 1
        return q

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if item >= self._len:
            raise IndexError(str(item))
        return self._list[item]

    def __str__(self):
        return 'Queue({})'.format(', '.join(str(x) for x in self))

    def __repr__(self):
        return 'Queue({})'.format(', '.join(repr(x) for x in self))
