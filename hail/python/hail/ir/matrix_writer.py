import abc
import json
from hail.expr.types import hail_type
from ..typecheck import typecheck_method, nullable, dictof, sequenceof
from ..utils.misc import escape_str
from .export_type import ExportType


class MatrixWriter(object):
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class MatrixNativeWriter(MatrixWriter):
    @typecheck_method(
        path=str,
        overwrite=bool,
        stage_locally=bool,
        codec_spec=nullable(str),
        partitions=nullable(str),
        partitions_type=nullable(hail_type),
    )
    def __init__(self, path, overwrite, stage_locally, codec_spec, partitions, partitions_type):
        self.path = path
        self.overwrite = overwrite
        self.stage_locally = stage_locally
        self.codec_spec = codec_spec
        self.partitions = partitions
        self.partitions_type = partitions_type

    def render(self):
        writer = {
            'name': 'MatrixNativeWriter',
            'path': self.path,
            'overwrite': self.overwrite,
            'stageLocally': self.stage_locally,
            'codecSpecJSONStr': self.codec_spec,
            'partitions': self.partitions,
            'partitionsTypeStr': self.partitions_type._parsable_string() if self.partitions_type is not None else None,
        }
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return (
            isinstance(other, MatrixNativeWriter)
            and other.path == self.path
            and other.overwrite == self.overwrite
            and other.stage_locally == self.stage_locally
            and other.codec_spec == self.codec_spec
            and other.partitions == self.partitions
            and other.partitions_type == self.partitions_type
        )


class MatrixVCFWriter(MatrixWriter):
    @typecheck_method(
        path=str,
        append=nullable(str),
        export_type=ExportType.checker,
        metadata=nullable(dictof(str, dictof(str, dictof(str, str)))),
        tabix=bool,
    )
    def __init__(self, path, append, export_type, metadata, tabix):
        self.path = path
        self.append = append
        self.export_type = export_type
        self.metadata = metadata
        self.tabix = tabix

    def render(self):
        writer = {
            'name': 'MatrixVCFWriter',
            'path': self.path,
            'append': self.append,
            'exportType': self.export_type,
            'metadata': self.metadata,
            'tabix': self.tabix,
        }
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return (
            isinstance(other, MatrixVCFWriter)
            and other.path == self.path
            and other.append == self.append
            and other.export_type == self.export_type
            and other.metadata == self.metadata
            and other.tabix == self.tabix
        )


class MatrixGENWriter(MatrixWriter):
    @typecheck_method(path=str, precision=int)
    def __init__(self, path, precision):
        self.path = path
        self.precision = precision

    def render(self):
        writer = {'name': 'MatrixGENWriter', 'path': self.path, 'precision': self.precision}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, MatrixGENWriter) and other.path == self.path and other.precision == self.precision


class MatrixBGENWriter(MatrixWriter):
    @typecheck_method(path=str, export_type=ExportType.checker, compression_codec=str)
    def __init__(self, path, export_type, compression_codec):
        self.path = path
        self.export_type = export_type
        self.compression_codec = compression_codec

    def render(self):
        writer = {
            'name': 'MatrixBGENWriter',
            'path': self.path,
            'exportType': self.export_type,
            'compressionCodec': self.compression_codec,
        }
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return (
            isinstance(other, MatrixBGENWriter)
            and other.path == self.path
            and other.export_type == self.export_type
            and other.compression_codec == self.compression_codec
        )


class MatrixPLINKWriter(MatrixWriter):
    @typecheck_method(path=str)
    def __init__(self, path):
        self.path = path

    def render(self):
        writer = {'name': 'MatrixPLINKWriter', 'path': self.path}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, MatrixPLINKWriter) and other.path == self.path


class MatrixBlockMatrixWriter(MatrixWriter):
    @typecheck_method(path=str, overwrite=bool, entry_field=str, block_size=int)
    def __init__(self, path, overwrite, entry_field, block_size):
        self.path = path
        self.overwrite = overwrite
        self.entry_field = entry_field
        self.block_size = block_size

    def render(self):
        writer = {
            'name': 'MatrixBlockMatrixWriter',
            'path': self.path,
            'overwrite': self.overwrite,
            'entryField': self.entry_field,
            'blockSize': self.block_size,
        }
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return (
            isinstance(other, MatrixBlockMatrixWriter)
            and other.path == self.path
            and other.overwrite == self.overwrite
            and other.entry_field == self.entry_field
            and other.block_size == self.block_size
        )


class MatrixNativeMultiWriter(object):
    @typecheck_method(paths=sequenceof(str), overwrite=bool, stage_locally=bool, codec_spec=nullable(str))
    def __init__(self, paths, overwrite, stage_locally, codec_spec):
        self.paths = paths
        self.overwrite = overwrite
        self.stage_locally = stage_locally
        self.codec_spec = codec_spec

    def render(self):
        writer = {
            'name': 'MatrixNativeMultiWriter',
            'paths': list(self.paths),
            'overwrite': self.overwrite,
            'stageLocally': self.stage_locally,
            'codecSpecJSONStr': self.codec_spec,
        }
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return (
            isinstance(other, MatrixNativeMultiWriter)
            and other.paths == self.paths
            and other.overwrite == self.overwrite
            and other.stage_locally == self.stage_locally
            and other.codec_spec == self.codec_spec
        )
