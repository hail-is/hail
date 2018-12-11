import abc
import json

from ..typecheck import *
from ..utils import wrap_to_list
from ..utils.java import escape_str
from ..genetics.reference_genome import reference_genome_type


class MatrixReader(object):
    @abc.abstractmethod
    def render(self, r):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class MatrixNativeReader(MatrixReader):
    @typecheck_method(path=str)
    def __init__(self, path):
        self.path = path

    def render(self, r):
        reader = {'name': 'MatrixNativeReader',
                  'path': self.path}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, MatrixNativeReader) and \
               other.path == self.path


class MatrixRangeReader(MatrixReader):
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      n_partitions=nullable(int))
    def __init__(self, n_rows, n_cols, n_partitions):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_partitions = n_partitions

    def render(self, r):
        reader = {'name': 'MatrixRangeReader',
                  'nRows': self.n_rows,
                  'nCols': self.n_cols,
                  'nPartitions': self.n_partitions}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, MatrixRangeReader) and \
               other.n_rows == self.n_rows and \
               other.n_cols == self.n_cols and \
               other.n_partitions == self.n_partitions


class MatrixVCFReader(MatrixReader):
    @typecheck_method(path=oneof(str, sequenceof(str)),
                      call_fields=oneof(str, sequenceof(str)),
                      header_file=nullable(str),
                      min_partitions=nullable(int),
                      reference_genome=nullable(reference_genome_type),
                      contig_recoding=nullable(dictof(str, str)),
                      array_elements_required=bool,
                      skip_invalid_loci=bool,
                      force_bgz=bool,
                      force_gz=bool)
    def __init__(self,
                 path,
                 call_fields,
                 header_file,
                 min_partitions,
                 reference_genome,
                 contig_recoding,
                 array_elements_required,
                 skip_invalid_loci,
                 force_bgz,
                 force_gz):
        self.path = wrap_to_list(path)
        self.header_file = header_file
        self.min_partitions = min_partitions
        self.call_fields = wrap_to_list(call_fields)
        self.reference_genome = reference_genome
        self.contig_recoding = contig_recoding
        self.array_elements_required = array_elements_required
        self.skip_invalid_loci = skip_invalid_loci
        self.force_gz = force_gz
        self.force_bgz = force_bgz

    def render(self, r):
        reader = {'name': 'MatrixVCFReader',
                  'files': self.path,
                  'callFields': self.call_fields,
                  'headerFile': self.header_file,
                  'minPartitions': self.min_partitions,
                  'rg': self.reference_genome.name if self.reference_genome else None,
                  'contigRecoding': self.contig_recoding if self.contig_recoding else {},
                  'arrayElementsRequired': self.array_elements_required,
                  'skipInvalidLoci': self.skip_invalid_loci,
                  'gzAsBGZ': self.force_bgz,
                  'forceGZ': self.force_gz}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, MatrixVCFReader) and \
               other.path == self.path and \
               other.call_fields == self.call_fields and \
               other.header_file == self.header_file and \
               other.min_partitions == self.min_partitions and \
               other.reference_genome == self.reference_genome and \
               other.contig_recoding == self.contig_recoding and \
               other.array_elements_required == self.array_elements_required and \
               other.skip_invalid_loci == self.skip_invalid_loci and \
               other.force_bgz == self.force_bgz and \
               other.force_gz == self.force_gz


class MatrixBGENReader(MatrixReader):
    @typecheck_method(path=oneof(str, sequenceof(str)),
                      sample_file=nullable(str),
                      index_file_map=nullable(dictof(str, str)),
                      n_partitions=nullable(int),
                      block_size=nullable(int),
                      included_variants=nullable(anytype))
    def __init__(self, path, sample_file, index_file_map, n_partitions, block_size, included_variants):
        self.path = wrap_to_list(path)
        self.sample_file = sample_file
        self.index_file_map = index_file_map if index_file_map else {}
        self.n_partitions = n_partitions
        self.block_size = block_size

        from hail.table import Table
        if included_variants is not None:
            assert(isinstance(included_variants, Table))
        self.included_variants = included_variants

    def render(self, r):
        reader = {'name': 'MatrixBGENReader',
                  'files': self.path,
                  'sampleFile': self.sample_file,
                  'indexFileMap': self.index_file_map,
                  'nPartitions': self.n_partitions,
                  'blockSizeInMB': self.block_size,
                  'includedVariants': r(self.included_variants._tir) if self.included_variants else None
                  }
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, MatrixBGENReader) and \
               other.path == self.path and \
               other.sample_file == self.sample_file and \
               other.index_file_map == self.index_file_map and \
               other.block_size == self.block_size and \
               other.included_variants == self.included_variants
