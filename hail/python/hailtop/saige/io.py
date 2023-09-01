import abc
import copy
from typing import Dict, List, Optional, Set, Tuple

import hailtop.batch as hb
from hailtop.batch import Batch, Job, Resource, ResourceGroup
import hail as hl

from .config import SaigeMatrixTableDataExtractorConfig, SaigeMatrixTableToBgenConfig, SaigeMatrixTableToVcfConfig


class SaigeResource(abc.ABC):
    @abc.abstractmethod
    def to_resource(self, b: Batch) -> hb.Resource:
        pass


class BgenInputFile(SaigeResource):
    @staticmethod
    def from_root_path(root: str) -> 'BgenInputFile':
        bgen = f'{root}.bgen'
        bgen_bgi = f'{root}.bgen.bgi'
        sample = f'{root}.sample'
        return BgenInputFile(bgen=bgen, bgen_bgi=bgen_bgi, sample=sample)

    def __init__(self, *, bgen: str, bgen_bgi: str, sample: str):
        self.bgen = bgen
        self.bgen_bgi = bgen_bgi
        self.sample = sample

    def to_resource(self, b: Batch) -> hb.Resource:
        return b.read_input_group(**{'bgen': self.bgen, 'bgen.bgi': self.bgen_bgi, 'sample': self.sample})


class VCFInputFile(SaigeResource):
    @staticmethod
    def from_root_path(root: str) -> 'VCFInputFile':
        vcf_gz = f'{root}.vcf.gz'
        vcf_gz_tbi = f'{root}.vcf.gz.tbi'
        return VCFInputFile(vcf_gz=vcf_gz, vcf_gz_tbi=vcf_gz_tbi)

    def __init__(self, *, vcf_gz: str, vcf_gz_tbi: str):
        self.vcf_gz = vcf_gz
        self.vcf_gz_tbi = vcf_gz_tbi

    def to_resource(self, b: Batch) -> hb.Resource:
        return b.read_input_group(**{'vcf.gz': self.vcf_gz, 'vcf.gz.tbi': self.vcf_gz_tbi})


class PlinkInputFile(SaigeResource):
    @staticmethod
    def from_root_path(root: str) -> 'PlinkInputFile':
        bed = f'{root}.bed'
        bim = f'{root}.bim'
        fam = f'{root}.fam'
        return PlinkInputFile(bed=bed, bim=bim, fam=fam)

    def __init__(self, *, bed: str, bim: str, fam: str):
        self.bed = bed
        self.bim = bim
        self.fam = fam

    def to_resource(self, b: Batch) -> hb.ResourceGroup:
        return b.read_input_group(bed=self.bed, bim=self.bim, fam=self.fam)


class SaigeSparseGRMResource:
    def __init__(self, j: hb.Job, mtx_identifier: str):
        self.j = j
        self.mtx_identifier = mtx_identifier
        self.resource = self._to_resource(j)

        self.grm_identifier = self.mtx_identifier
        self.sample_ids_identifier = f'{self.mtx_identifier}.sampleIDs.txt'

    def _to_resource(self, j: hb.Job) -> hb.ResourceGroup:
        j.declare_resource_group(
            sparse_grm={
                f'{self.grm_identifier}': f'{{root}}{self.grm_identifier}',
                f'{self.sample_ids_identifier}': f'{{root}}{self.sample_ids_identifier}',
            }
        )
        return j.sparse_grm

    @property
    def sample_ids(self) -> hb.ResourceFile:
        return self.resource[self.sample_ids_identifier]

    @property
    def sparse_grm(self) -> hb.ResourceFile:
        return self.resource[self.grm_identifier]


def generate_groups(gene_ht: hl.Table,
                    *,
                    gene: Optional[str] = None,
                    gene_ht_interval: Optional[str] = None,
                    interval: Optional[str] = None,
                    groups: Optional[Set[str]] = None) -> Tuple[hl.Table, List[hl.Interval]]:
    if groups is None:
        groups = {'pLoF', 'missense|LC', 'synonymous'}

    if gene is not None:
        gene_ht = gene_ht.filter(gene_ht.gene_symbol == gene)
        intervals = gene_ht.aggregate(hl.agg.take(gene_ht.interval, 1), _localize=False)
    else:
        gene_ht = hl.filter_intervals(gene_ht, [hl.parse_locus_interval(gene_ht_interval)])
        intervals = [hl.parse_locus_interval(interval)]

    gene_ht = (gene_ht.filter(hl.set(groups).contains(gene_ht.annotation)))

    groups_ht = (
        gene_ht.select(
            group=gene_ht.gene_id + '_' + gene_ht.gene_symbol + '_' + gene_ht.annotation + hl.if_else(gene_ht.common_variant, '_' + gene_ht.variants[0], ''),
            variant=hl.delimit(gene_ht.variants, '\t')
        )
            .key_by()
            .drop('start')
     )

    return (groups_ht, intervals)

        # if config.common_variants_only:
        #     gene_ht = gene_ht.filter(gene_ht.common_variant)




def extract_vcf_from_mt(mt: hl.MatrixTable, config: SaigeMatrixTableDataExtractorConfig):
        # TODO: possible minor optimization: filter output VCF to only variants in `gene_ht.variants`

    if config.group_file_only:
        return

    if not args.no_adj:
        mt = mt.filter_entries(mt.adj)

    mt = hl.filter_intervals(mt, interval)

    if not args.input_bgen:
        mt = mt.select_entries('GT')
        mt = mt.filter_rows(hl.agg.count_where(mt.GT.is_non_ref()) > 0)
    mt = mt.annotate_rows(rsid=mt.locus.contig + ':' + hl.str(mt.locus.position) + '_' + mt.alleles[0] + '/' + mt.alleles[1])

    if args.callrate_filter:
        mt = mt.filter_rows(hl.agg.fraction(hl.is_defined(mt.GT)) >= args.callrate_filter)

    if args.export_bgen:
        if not args.input_bgen:
            mt = mt.annotate_entries(GT=hl.if_else(mt.GT.is_haploid(), hl.call(mt.GT[0], mt.GT[0]), mt.GT))
            mt = gt_to_gp(mt)
            mt = impute_missing_gp(mt, mean_impute=args.mean_impute_missing)
        hl.export_bgen(mt, args.output_file, gp=mt.GP, varid=mt.rsid)
    else:
        mt = mt.annotate_entries(GT=hl.or_else(mt.GT, hl.call(0, 0)))
        # Note: no mean-imputation for VCF
        hl.export_vcf(mt, args.output_file)


    attributes = copy.deepcopy(config.attributes or {})
    attributes.update(interval=config.interval)

    extract_j: Job = p.new_job(name=config.name, attributes=attributes)

    (extract_j
     .image(config.docker_image)
     .cpu(config.n_threads)
     .storage(config.storage)
     )

    if export_bgen:

    else:
        extract_j.declare_resource_group(out={'vcf.gz': f'{{root}}.vcf.gz',
                                                 'vcf.gz.tbi': f'{{root}}.vcf.gz.tbi'})

    output_file = f'{extract_j.bgz}.bgz' if not export_bgen else extract_j.out
    command = f"""set -o pipefail; PYSPARK_SUBMIT_ARGS="--conf spark.driver.memory={int(3 * n_threads)}g pyspark-shell"
    python3 {SCRIPT_DIR}/extract_vcf_from_mt.py
    --load_module {module}
    {"--additional_args " + additional_args if additional_args else ''}
    {"--gene " + gene if gene else ""}
    {"--interval " + interval if interval else ""}
    {"--gene_ht_interval " + gene_ht_interval if gene_ht_interval else ""}
    --groups "{','.join(groups)}"
    --reference {reference} --n_threads {n_threads}
    {"--gene_map_ht_path " + gene_map_ht_path if gene_map_ht_path else ""} 
    {"--callrate_filter " + str(callrate_filter) if callrate_filter else ""} 
    {"--export_bgen" if export_bgen else ""}
    {"--input_bgen" if input_dosage else ""}
    {"" if set_missing_to_hom_ref else "--mean_impute_missing"}
    {"" if adj else "--no_adj"} 
    {"--group_output_file " + extract_j.group_file if gene_map_ht_path else ""}
    --output_file {output_file} | tee {extract_j.stdout}
    ;""".replace('\n', ' ')

    if export_bgen:
        command += f'\n/bgen_v1.1.4-Ubuntu16.04-x86_64/bgenix -g {extract_j.out.bgen} -index -clobber'
    else:
        command += f'\nmv {extract_j.bgz}.bgz {extract_j.out["vcf.gz"]}; tabix {extract_j.out["vcf.gz"]};'
    extract_j.command(command.replace('\n', ' '))

    activate_service_account(extract_j)
    p.write_output(extract_j.out, output_root)
    if gene_map_ht_path:
        p.write_output(extract_j.group_file, f'{output_root}.gene.txt')
    p.write_output(extract_j.stdout, f'{output_root}.log')
    return extract_j