import abc
import json

import avro.schema

import hail as hl

from hail.ir.utils import make_filter_and_replace, default_row_uid
from hail.typecheck import typecheck_method, sequenceof, nullable, anytype, oneof
from hail.utils.misc import escape_str

from .utils import impute_type_of_partition_interval_array


class TableReader(object):
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class TableNativeReader(TableReader):
    @typecheck_method(path=str,
                      intervals=nullable(sequenceof(anytype)),
                      filter_intervals=bool)
    def __init__(self, path, intervals, filter_intervals):
        self.path = path
        self.filter_intervals = filter_intervals
        self.intervals, self._interval_type = impute_type_of_partition_interval_array(intervals)

    def render(self):
        reader = {'name': 'TableNativeReader',
                  'path': self.path}
        if self.intervals is not None:
            assert self._interval_type is not None
            reader['options'] = {
                'name': 'NativeReaderOptions',
                'intervals': self._interval_type._convert_to_json(self.intervals),
                'intervalPointType': self._interval_type.element_type.point_type._parsable_string(),
                'filterIntervals': self.filter_intervals
            }
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TableNativeReader) and \
            other.path == self.path and \
            other.intervals == self.intervals and \
            other.filter_intervals == self.filter_intervals


class TextTableReader(TableReader):
    def __init__(self, paths, min_partitions, types, comment,
                 delimiter, missing, no_header, quote,
                 skip_blank_lines, force_bgz, filter, find_replace,
                 force_gz, source_file_field):
        self.config = {
            'files': paths,
            'typeMapStr': {f: t._parsable_string() for f, t in types.items()},
            'comment': comment,
            'separator': delimiter,
            'missing': missing,
            'hasHeader': not no_header,
            'nPartitionsOpt': min_partitions,
            'quoteStr': quote,
            'skipBlankLines': skip_blank_lines,
            'forceBGZ': force_bgz,
            'filterAndReplace': make_filter_and_replace(filter, find_replace),
            'forceGZ': force_gz,
            'sourceFileField': source_file_field
        }

    def render(self):
        reader = dict(self.config)
        reader['name'] = 'TextTableReader'
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TextTableReader) and \
            other.config == self.config


class StringTableReader(TableReader):
    @typecheck_method(paths=oneof(str, sequenceof(str)), min_partitions=nullable(int), force_bgz=bool,
                      force=bool, file_per_partition=bool)
    def __init__(self, paths, min_partitions, force_bgz, force, file_per_partition):
        self.paths = paths
        self.min_partitions = min_partitions
        self.force_bgz = force_bgz
        self.force = force
        self.file_per_partition = file_per_partition

    def render(self):
        reader = {'name': 'StringTableReader',
                  'files': self.paths,
                  'minPartitions': self.min_partitions,
                  'forceBGZ': self.force_bgz,
                  'forceGZ': self.force,
                  'filePerPartition': self.file_per_partition}

        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, StringTableReader) and \
            other.path == self.path and \
            other.min_partitions == self.min_partitions and \
            other.force_bgz == self.force_bgz and \
            other.force == self.force and \
            other.file_per_partition == self.file_per_partition


class TableFromBlockMatrixNativeReader(TableReader):
    @typecheck_method(path=str, n_partitions=nullable(int), maximum_cache_memory_in_bytes=nullable(int))
    def __init__(self, path, n_partitions, maximum_cache_memory_in_bytes):
        self.path = path
        self.n_partitions = n_partitions
        self.maximum_cache_memory_in_bytes = maximum_cache_memory_in_bytes

    def render(self):
        reader = {'name': 'TableFromBlockMatrixNativeReader',
                  'path': self.path,
                  'nPartitions': self.n_partitions,
                  'maximumCacheMemoryInBytes': self.maximum_cache_memory_in_bytes}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, TableFromBlockMatrixNativeReader) and \
            other.path == self.path and \
            other.n_partitions == self.n_partitions and \
            other.maximum_cache_memory_in_bytes == self.maximum_cache_memory_in_bytes


class AvroTableReader(TableReader):
    @typecheck_method(schema=avro.schema.Schema,
                      paths=sequenceof(str),
                      key=nullable(sequenceof(str)),
                      intervals=nullable(sequenceof(anytype)))
    def __init__(self, schema, paths, key, intervals):
        assert (key is None) == (intervals is None)
        self.schema = schema
        self.paths = paths
        self.key = key

        if intervals is not None:
            t = hl.expr.impute_type(intervals)
            if not isinstance(t, hl.tarray) and not isinstance(t.element_type, hl.tinterval):
                raise TypeError("'intervals' must be an array of tintervals")
            pt = t.element_type.point_type
            if isinstance(pt, hl.tstruct):
                self._interval_type = t
            else:
                self._interval_type = hl.tarray(hl.tinterval(hl.tstruct(__point=pt)))

        if intervals is not None and t != self._interval_type:
            self.intervals = [hl.Interval(hl.Struct(__point=i.start),
                                          hl.Struct(__point=i.end),
                                          i.includes_start,
                                          i.includes_end) for i in intervals]
        else:
            self.intervals = intervals

    def render(self):
        reader = {'name': 'AvroTableReader',
                  'partitionReader': {'name': 'AvroPartitionReader',
                                      'schema': self.schema.to_json(),
                                      'uidFieldName': default_row_uid},
                  'paths': self.paths}
        if self.key is not None:
            assert self.intervals is not None
            assert self._interval_type is not None
            reader['unsafeOptions'] = {
                'name': 'UnsafeAvroTableReaderOptions',
                'key': self.key,
                'intervals': self._interval_type._convert_to_json(self.intervals),
                'intervalPointType': self._interval_type.element_type.point_type._parsable_string(),
            }
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, AvroTableReader) and \
            other.schema == self.schema and \
            other.paths == self.paths and \
            other.key == self.key and \
            other.intervals == self.intervals


class GoogleSheetReader(TableReader):
    @typecheck_method(spreadsheetID=oneof(str, sequenceof(str)), sheetname=str)
    def __init__(self, spreadsheetID, sheetname):
        self.spreadsheetID = spreadsheetID
        self.sheetname = sheetname

    def render(self):
        reader = {'name': 'GoogleSheetReader',
                  'spreadsheetID': self.spreadsheetID,
                  'sheetname': self.sheetname}

        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, GoogleSheetReader) and \
            other.spreadsheetID == self.spreadsheetID and \
            other.sheetname == self.sheetname
