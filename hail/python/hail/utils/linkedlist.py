from collections import Iterable, Iterator

class ListIterator(Iterator):
    def __init__(self, node):
        self.node = node

    def __next__(self):
        if self.node is None:
            raise StopIteration
        else:
            x = self.node.value
            self.node = self.node.prev
            return x


class ListNode(object):
    def __init__(self, value, prev):
        self.value = value
        self.prev = prev

class LinkedList(Iterable):
    def __init__(self, type):
        self.type = type
        self.node = None

    def push(self, *xs):
        l = LinkedList.__new__(LinkedList)
        l.type = self.type
        l.node = self.node
        for x in xs:
            if not isinstance(x, self.type):
                raise TypeError("Expected type '{}', found type '{}': {}".format(self.type, type(x).__class__, x))
            l.node = ListNode(x, l.node)
        return l

    def empty(self):
        return self.node is None

    def __iter__(self):
        return ListIterator(self.node)

    def __str__(self):
        return 'List({})'.format(', '.join(str(x) for x in self))

    def __repr__(self):
        return 'List({})'.format(', '.join(repr(x) for x in self))

    def __eq__(self, other):
        return isinstance(other, LinkedList) and list(self) == list(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return not self.empty()

    def __len__(self):
        l = 0
        for _ in self:
            l += 1
        return l