from __future__ import print_function  # Python 2 and 3 print compatibility

from pyspark import SparkContext
from pyspark.sql import SQLContext

from hail2.dataset import VariantDataset
from hail.typ import Type
from hail.java import *
from hail2.keytable import KeyTable
from hail.stats import UniformDist, TruncatedBetaDist, BetaDist
from hail.utils import wrap_to_list
from hail.history import *
from hail.typecheck import *
from hail.representation import GenomeReference
import hail


class HailContext(HistoryMixin):
    @record_init
    @typecheck_method(sc=nullable(SparkContext),
                      app_name=strlike,
                      master=nullable(strlike),
                      local=strlike,
                      log=strlike,
                      quiet=bool,
                      append=bool,
                      min_block_size=integral,
                      branching_factor=integral,
                      tmp_dir=strlike)
    def __init__(self, sc=None, app_name="Hail", master=None, local='local[*]',
                 log='hail.log', quiet=False, append=False,
                 min_block_size=1, branching_factor=50, tmp_dir='/tmp'):

        self.hc1 = hail.HailContext(sc, app_name, master, local, log, quiet, append, min_block_size, branching_factor, tmp_dir)
        self._counter = 0

    @staticmethod
    def get_running():
        hail.HailContext.get_running()

    @property
    def version(self):
        return self.hc1.version

    @handle_py4j
    @typecheck_method(regex=strlike,
                      path=oneof(strlike, listof(strlike)),
                      max_count=integral)
    def grep(self, regex, path, max_count=100):
        self.hc1.grep(regex, jindexed_seq_args(path), max_count)

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      tolerance=numeric,
                      sample_file=nullable(strlike),
                      min_partitions=nullable(integral),
                      reference_genome=nullable(GenomeReference))
    def import_bgen(self, path, tolerance=0.2, sample_file=None, min_partitions=None, reference_genome=None):
        return self.hc1.import_bgen(path, tolerance, sample_file, min_partitions, reference_genome).to_hail2()

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      sample_file=nullable(strlike),
                      tolerance=numeric,
                      min_partitions=nullable(integral),
                      chromosome=nullable(strlike),
                      reference_genome=nullable(GenomeReference))
    def import_gen(self, path, sample_file=None, tolerance=0.2, min_partitions=None, chromosome=None, reference_genome=None):
        return self.hc1.import_gen(path, sample_file, tolerance, min_partitions, chromosome, reference_genome).to_hail2()

    @handle_py4j
    @record_method
    @typecheck_method(paths=oneof(strlike, listof(strlike)),
                      key=oneof(strlike, listof(strlike)),
                      min_partitions=nullable(int),
                      impute=bool,
                      no_header=bool,
                      comment=nullable(strlike),
                      delimiter=strlike,
                      missing=strlike,
                      types=dictof(strlike, Type),
                      quote=nullable(char),
                      reference_genome=nullable(GenomeReference))
    def import_table(self, paths, key=[], min_partitions=None, impute=False, no_header=False,
                     comment=None, delimiter="\t", missing="NA", types={}, quote=None, reference_genome=None):
        return self.hc1.import_table(paths, key, min_partitions, impute, no_header, comment,
                                     delimiter, missing, types, quote, reference_genome).to_hail2()

    @handle_py4j
    @record_method
    @typecheck_method(bed=strlike,
                      bim=strlike,
                      fam=strlike,
                      min_partitions=nullable(integral),
                      delimiter=strlike,
                      missing=strlike,
                      quant_pheno=bool,
                      a2_reference=bool,
                      reference_genome=nullable(GenomeReference))
    def import_plink(self, bed, bim, fam, min_partitions=None, delimiter='\\\\s+',
                     missing='NA', quant_pheno=False, a2_reference=True, reference_genome=None):
        return self.hc1.import_plink(bed, bim, fam, min_partitions, delimiter,
                                     missing, quant_pheno, a2_reference, reference_genome).to_hail2()

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      drop_samples=bool,
                      drop_variants=bool)
    def read(self, path, drop_samples=False, drop_variants=False):
        return self.hc1.read(path, drop_samples, drop_variants).to_hail2()

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      force=bool,
                      force_bgz=bool,
                      header_file=nullable(strlike),
                      min_partitions=nullable(integral),
                      drop_samples=bool,
                      call_fields=oneof(strlike, listof(strlike)),
                      reference_genome=nullable(GenomeReference))
    def import_vcf(self, path, force=False, force_bgz=False, header_file=None, min_partitions=None,
                   drop_samples=False, call_fields=[], reference_genome=None):
        return self.hc1.import_vcf(path, force, force_bgz, header_file, min_partitions,
                                   drop_samples, call_fields, reference_genome).to_hail2()


    @handle_py4j
    @typecheck_method(path=oneof(strlike, listof(strlike)))
    def index_bgen(self, path):
        self.hc1.index_bgen(path)

    @handle_py4j
    @record_method
    @typecheck_method(populations=integral,
                      samples=integral,
                      variants=integral,
                      num_partitions=nullable(integral),
                      pop_dist=nullable(listof(numeric)),
                      fst=nullable(listof(numeric)),
                      af_dist=oneof(UniformDist, BetaDist, TruncatedBetaDist),
                      seed=integral,
                      reference_genome=nullable(GenomeReference))
    def balding_nichols_model(self, populations, samples, variants, num_partitions=None,
                              pop_dist=None, fst=None, af_dist=UniformDist(0.1, 0.9),
                              seed=0, reference_genome=None):
        return self.hc1.balding_nichols_model(populations, samples, variants, num_partitions,
                                              pop_dist, fst, af_dist, seed, reference_genome).to_hail2()

    @handle_py4j
    @typecheck_method(expr=strlike)
    def eval_expr_typed(self, expr):
        self.hc1.eval_expr_typed(expr)

    @handle_py4j
    @typecheck_method(expr=strlike)
    def eval_expr(self, expr):
        self.hc1.eval_expr(expr)

    def stop(self):
        self.hc1.stop()

    @handle_py4j
    @record_method
    @typecheck_method(path=strlike)
    def read_table(self, path):
        return self.hc1.read_table(path)

    def _get_unique_id(self):
        self._counter += 1
        return "__uid_{}".format(self._counter)
