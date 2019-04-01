import abc
import json

from hail.ir.utils import make_filter_and_replace
from hail.typecheck import *
from hail.utils.java import escape_str


class TableReader(object):
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class TableNativeReader(TableReader):
    @typecheck_method(path=str)
    def __init__(self, path):
        self.path = path

    def render(self):
        reader = {'name': 'TableNativeReader',
                  'path': self.path}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TableNativeReader) and \
               other.path == self.path

class TextTableReader(TableReader):
    def __init__(self, paths, min_partitions, types, comment,
                 delimiter, missing, no_header, impute, quote,
                 skip_blank_lines, force_bgz, filter, find_replace,
                 force_gz):
        self.config = {
            'files': paths,
            'typeMapStr': {f: t._parsable_string() for f, t in types.items()},
            'comment': comment,
            'separator': delimiter,
            'missing': missing,
            'noHeader': no_header,
            'impute': impute,
            'nPartitionsOpt': min_partitions,
            'quoteStr': quote,
            'skipBlankLines': skip_blank_lines,
            'forceBGZ': force_bgz,
            'filterAndReplace': make_filter_and_replace(filter, find_replace),
            'forceGZ': force_gz
        }

    def render(self):
        reader = {'name': 'TextTableReader',
                  'options': self.config}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TextTableReader) and \
               other.config == self.config


class TableFromBlockMatrixNativeReader(TableReader):
    @typecheck_method(path=str, n_partitions=nullable(int))
    def __init__(self, path, n_partitions):
        self.path = path
        self.n_partitions = n_partitions

    def render(self):
        reader = {'name': 'TableFromBlockMatrixNativeReader',
                  'path': self.path,
                  'nPartitions': self.n_partitions}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TableFromBlockMatrixNativeReader) and \
               other.path == self.path and \
               other.n_partitions == self.n_partitions
