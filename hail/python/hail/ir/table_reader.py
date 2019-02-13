import abc
import json

from hail.typecheck import *
from hail.utils.java import escape_str
from hail.ir.utils import make_filter_and_replace

class TableReader(object):
    @abc.abstractmethod
    def render(self, r):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class TableNativeReader(TableReader):
    @typecheck_method(path=str)
    def __init__(self, path):
        self.path = path

    def render(self, r):
        reader = {'name': 'TableNativeReader',
                  'path': self.path}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TableNativeReader) and \
               other.path == self.path

class TextTableReader(TableReader):
    def __init__(self, paths, min_partitions, types, comment,
                 delimiter, missing, no_header, impute, quote,
                 skip_blank_lines, force_bgz, filter, find_replace):
        self.config = {
            'files': paths,
            'typeMapStr': {f: t._parsable_string() for f, t in types.items()},
            'comment': comment,
            'separator': delimiter,
            'missing': missing,
            'noHeader': no_header,
            'impute': impute,
            'nPartitions': min_partitions,
            'quoteStr': quote,
            'skipBlankLines': skip_blank_lines,
            'forceBGZ': force_bgz,
            'filterAndReplace': make_filter_and_replace(filter, find_replace)
        }

    def render(self, r):
        reader = {'name': 'TextTableReader',
                  'options': self.config}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TextTableReader) and \
               other.config == self.config
