import abc
from ..fs.fs import FS


class Backend(abc.ABC):
    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def execute(self, ir, timed=False):
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
    def index_bgen(self, files, index_file_map, rg, contig_recoding, skip_invalid_loci):
        pass

    @abc.abstractmethod
    def import_fam(self, path: str, quant_pheno: bool, delimiter: str, missing: str):
        pass

    def persist_table(self, t, storage_level):
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
    def register_ir_function(self, name, type_parameters, argument_names, argument_types, return_type, body):
        pass

    @abc.abstractmethod
    def persist_ir(self, ir):
        pass
