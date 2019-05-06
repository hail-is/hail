import abc
import json
from ..typecheck import *
from ..utils.java import escape_str


class MatrixWriter(object):
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class MatrixNativeWriter(MatrixWriter):
    @typecheck_method(path=str,
                      overwrite=bool,
                      stage_locally=bool,
                      codec_spec=nullable(str))
    def __init__(self, path, overwrite, stage_locally, codec_spec):
        self.path = path
        self.overwrite = overwrite
        self.stage_locally = stage_locally
        self.codec_spec = codec_spec

    def render(self):
        writer = {'name': 'MatrixNativeWriter',
                  'path': self.path,
                  'overwrite': self.overwrite,
                  'stageLocally': self.stage_locally,
                  'codecSpecJSONStr': self.codec_spec}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, MatrixNativeWriter) and \
               other.path == self.path and \
               other.overwrite == self.overwrite and \
               other.stage_locally == self.stage_locally and \
               other.codec_spec == self.codec_spec


class MatrixVCFWriter(MatrixWriter):
    @typecheck_method(path=str,
                      append=nullable(str),
                      export_type=int,
                      metadata=nullable(dictof(str, dictof(str, dictof(str, str)))))
    def __init__(self, path, append, export_type, metadata):
        self.path = path
        self.append = append
        self.export_type = export_type
        self.metadata = metadata

    def render(self):
        writer = {'name': 'MatrixVCFWriter',
                  'path': self.path,
                  'append': self.append,
                  'exportType': self.export_type,
                  'metadata': self.metadata}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, MatrixVCFWriter) and \
               other.path == self.path and \
               other.append == self.append and \
               other.export_type == self.export_type and \
               other.metadata == self.metadata


class MatrixGENWriter(MatrixWriter):
    @typecheck_method(path=str,
                      precision=int)
    def __init__(self, path, precision):
        self.path = path
        self.precision = precision

    def render(self):
        writer = {'name': 'MatrixGENWriter',
                  'path': self.path,
                  'precision': self.precision}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, MatrixGENWriter) and \
               other.path == self.path and \
               other.precision == self.precision


class MatrixPLINKWriter(MatrixWriter):
    @typecheck_method(path=str)
    def __init__(self, path):
        self.path = path

    def render(self):
        writer = {'name': 'MatrixPLINKWriter',
                  'path': self.path}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, MatrixPLINKWriter) and \
               other.path == self.path


class MatrixNativeMultiWriter(object):
    @typecheck_method(prefix=str,
                      overwrite=bool,
                      stage_locally=bool)
    def __init__(self, prefix, overwrite, stage_locally):
        self.prefix = prefix
        self.overwrite = overwrite
        self.stage_locally = stage_locally

    def render(self):
        writer = {'name': 'MatrixNativeMultiWriter',
                  'prefix': self.prefix,
                  'overwrite': self.overwrite,
                  'stageLocally': self.stage_locally}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, MatrixNativeMultiWriter) and \
               other.prefix == self.prefix and \
               other.overwrite == self.overwrite and \
               other.stage_locally == self.stage_locally
