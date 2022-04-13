from typing import Mapping, List, Union, Tuple, Dict, Optional, Any
import abc
from ..fs.fs import FS
from ..expr import Expression
from ..expr.types import HailType
from ..ir import BaseIR
from ..utils.java import FatalError, HailUserError


def fatal_error_from_java_error_triplet(short_message, expanded_message, error_id):
    from .. import __version__
    if error_id != -1:
        return FatalError(f'Error summary: {short_message}', error_id)
    return FatalError(f'''{short_message}

Java stack trace:
{expanded_message}
Hail version: {__version__}
Error summary: {short_message}''',
                      error_id)


class Backend(abc.ABC):
    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def execute(self, ir: BaseIR, timed: bool = False) -> Any:
        pass

    @abc.abstractmethod
    async def _async_execute(self, ir, timed=False):
        pass

    def execute_many(self, *irs, timed=False):
        from ..ir import MakeTuple  # pylint: disable=import-outside-toplevel
        return [self.execute(MakeTuple([ir]), timed=timed)[0] for ir in irs]

    @abc.abstractmethod
    async def _async_execute_many(self, *irs, timed=False):
        pass

    @abc.abstractmethod
    def value_type(self, ir):
        pass

    @abc.abstractmethod
    def table_type(self, tir):
        pass

    @abc.abstractmethod
    def matrix_type(self, mir):
        pass

    @abc.abstractmethod
    def add_reference(self, config):
        pass

    @abc.abstractmethod
    def load_references_from_dataset(self, path):
        pass

    @abc.abstractmethod
    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        pass

    @abc.abstractmethod
    def remove_reference(self, name):
        pass

    @abc.abstractmethod
    def get_reference(self, name):
        pass

    @abc.abstractmethod
    async def _async_get_reference(self, name):
        pass

    def get_references(self, names):
        return [self.get_reference(name) for name in names]

    @abc.abstractmethod
    async def _async_get_references(self, names):
        pass

    @abc.abstractmethod
    def add_sequence(self, name, fasta_file, index_file):
        pass

    @abc.abstractmethod
    def remove_sequence(self, name):
        pass

    @abc.abstractmethod
    def add_liftover(self, name, chain_file, dest_reference_genome):
        pass

    @abc.abstractmethod
    def remove_liftover(self, name, dest_reference_genome):
        pass

    @abc.abstractmethod
    def parse_vcf_metadata(self, path):
        pass

    @property
    @abc.abstractmethod
    def logger(self):
        pass

    @property
    @abc.abstractmethod
    def fs(self) -> FS:
        pass

    @abc.abstractmethod
    def index_bgen(self,
                   files: List[str],
                   index_file_map: Dict[str, str],
                   referenceGenomeName: Optional[str],
                   contig_recoding: Dict[str, str],
                   skip_invalid_loci: bool):
        pass

    @abc.abstractmethod
    def import_fam(self, path: str, quant_pheno: bool, delimiter: str, missing: str):
        pass

    def persist_table(self, t, storage_level):
        # FIXME: this can't possibly be right.
        return t

    def unpersist_table(self, t):
        return t

    def persist_matrix_table(self, mt, storage_level):
        return mt

    def unpersist_matrix_table(self, mt):
        return mt

    def unpersist_block_matrix(self, id):
        pass

    @abc.abstractmethod
    def register_ir_function(self,
                             name: str,
                             type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                             value_parameter_names: Union[Tuple[str, ...], List[str]],
                             value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                             return_type: HailType,
                             body: Expression):
        pass

    @abc.abstractmethod
    def persist_expression(self, expr: Expression) -> Expression:
        pass

    @abc.abstractmethod
    def set_flags(self, **flags: Mapping[str, str]):
        """Set Hail flags."""
        pass

    @abc.abstractmethod
    def get_flags(self, *flags) -> Mapping[str, str]:
        """Mapping of Hail flags."""
        pass

    @property
    @abc.abstractmethod
    def requires_lowering(self):
        pass

    def _handle_fatal_error_from_backend(self, err: FatalError, ir: BaseIR):
        if err._error_id is None:
            raise err

        error_sources = ir.base_search(lambda x: x._error_id == err._error_id)
        if len(error_sources) == 0:
            raise err

        better_stack_trace = error_sources[0]._stack_trace
        error_message = str(err)
        message_and_trace = (f'{error_message}\n'
                             '------------\n'
                             'Hail stack trace:\n'
                             f'{better_stack_trace}')
        raise HailUserError(message_and_trace) from None
