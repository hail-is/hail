import collections
import datetime
import typer
from typer import Option as Opt, Argument as Arg
import yaml

from dataclasses import asdict, dataclass, field, replace
from typing import Annotated as Ann, Any, Dict, List, Optional, Tuple, Union

import hail as hl
from hail.methods.qc import require_col_key_str, require_row_key_variant
import hailtop.batch as hb
import hailtop.fs as hfs

from .constants import SaigeAnalysisType, SaigeInputDataType
from .phenotype import Phenotypes
from .steps import PrepareInputsStep, SparseGRMStep, Step1NullGlmmStep, Step2SPAStep
from .variant_chunk import VariantChunks
from .utils import cast_dataclass_attributes

app = typer.Typer()


class DummyTemporaryDirectory:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@dataclass
class SaigeConfig:
    version: int = 1
    name: Optional[str] = 'saige'
    attributes: Optional[Dict[str, str]] = field(default_factory=collections.defaultdict, kw_only=True)
    prepare_inputs: PrepareInputsStep = field(default_factory=PrepareInputsStep, kw_only=True)
    sparse_grm: SparseGRMStep = field(default_factory=SparseGRMStep, kw_only=True)
    step1_null_glmm: Step1NullGlmmStep = field(default_factory=Step1NullGlmmStep, kw_only=True)
    step2_spa: Step2SPAStep = field(default_factory=Step2SPAStep, kw_only=True)

    def __post_init__(self):
        cast_dataclass_attributes(self)

    @staticmethod
    def from_yaml_file(file: str) -> 'SaigeConfig':
        with hfs.open(file, 'r') as f:
            config = yaml.safe_load(f)
        return SaigeConfig(**config)

    def to_yaml(self):
        return yaml.safe_dump(asdict(self))

    def update(self, updates: List[Tuple[str, str]]):
        changes: Dict[str, Any] = collections.defaultdict(dict)
        for name, value in updates:
            if '.' not in name:
                changes[name] = value
            else:
                step, var_name = name.split('.')
                changes[step][var_name] = value

        changes['prepare_inputs'] = replace(self.prepare_inputs, **changes.get('prepare_inputs', {}))
        changes['sparse_grm'] = replace(self.sparse_grm, **changes.get('sparse_grm', {}))
        changes['step1_null_glmm'] = replace(self.step1_null_glmm, **changes.get('step1_null_glmm', {}))
        changes['step2_spa'] = replace(self.step2_spa, **changes.get('step2_spa', {}))

        return replace(self, **changes)


class SAIGE:
    @staticmethod
    def from_config_file(file: str) -> 'SAIGE':
        config = SaigeConfig.from_yaml_file(file)
        return SAIGE(config)

    def __init__(self, config: Optional[SaigeConfig] = None):
        if config is None:
            config = SaigeConfig()
        self.config = config

    def _munge_inputs(
        self,
        *,
        mt: Union[str, hl.MatrixTable],
        checkpoint_dir: str,
        phenotypes: Union[List[str], Phenotypes],
        variant_chunks: Optional[Union[str, VariantChunks]],
    ) -> Tuple[hl.MatrixTable, str, Phenotypes, VariantChunks]:
        if isinstance(mt, hl.MatrixTable):
            mt_path = f'{checkpoint_dir}/input.mt'
            mt.checkpoint(mt_path)
        else:
            mt_path = mt
            mt = hl.read_matrix_table(mt_path)

        require_row_key_variant(mt, 'saige')
        require_col_key_str(mt, 'saige')

        if isinstance(phenotypes, list):
            mt_phenotypes = Phenotypes.from_matrix_table(mt, phenotypes)
            if len(phenotypes) != mt_phenotypes:
                raise ValueError('missing phenotypes from MatrixTable')
            phenotypes = mt_phenotypes

        if variant_chunks is None:
            variant_chunks = VariantChunks.from_matrix_table(mt)
        elif isinstance(variant_chunks, str):
            if variant_chunks.endswith('.bed'):
                variant_chunks = VariantChunks.from_bed(variant_chunks)
            else:
                variant_chunks = VariantChunks.from_locus_intervals(variant_chunks)
        else:
            assert isinstance(variant_chunks, VariantChunks)

        return (mt, mt_path, phenotypes, variant_chunks)

    def _dump_data_config_to_yaml(
        self,
        mt_path: str,
        output_dir: str,
        checkpoint_dir: str,
        phenotypes: Phenotypes,
        chunks: VariantChunks,
        input_data_type: SaigeInputDataType,
    ):
        data_config = {
            'mt_path': mt_path,
            'output_dir': output_dir,
            'checkpoint_dir': checkpoint_dir,
            'input_data_type': input_data_type,
            'phenotypes': [asdict(p) for p in phenotypes],
            'chunks': [c.to_dict() for c in chunks],
        }
        return yaml.safe_dump(data_config)

    def run_saige(
        self,
        *,
        mt: Union[str, hl.MatrixTable],
        phenotypes: Union[List[str], Phenotypes],
        covariates: List[str],
        output_dir: str,
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        variant_chunks: Optional[Union[str, VariantChunks]] = None,
        run_kwargs: Optional[dict] = None,
    ):
        if checkpoint_dir is None:
            checkpoint_dir = hl.TemporaryDirectory()
        else:
            checkpoint_dir = DummyTemporaryDirectory(checkpoint_dir)

        with checkpoint_dir:
            mt, mt_path, phenotypes, variant_chunks = self._munge_inputs(
                mt=mt, checkpoint_dir=checkpoint_dir.name, phenotypes=phenotypes, variant_chunks=variant_chunks
            )

            with hfs.open(f'{output_dir}/config.yaml', 'w') as f:
                f.write(self.config.to_yaml() + '\n')

            if b is None:
                b = hb.Batch(name=self.config.name, attributes=self.config.attributes)

            input_phenotypes, input_plink_data = self.config.prepare_inputs(b, mt, phenotypes, checkpoint_dir.name)

            user_id_col = list(mt.col_key)[0]

            if 'GP' in list(mt.entry):
                input_data_type = SaigeInputDataType.BGEN
            else:
                assert 'GT' in list(mt.entry)
                input_data_type = SaigeInputDataType.VCF

            with hfs.open(f'{output_dir}/data_config.yaml', 'w') as f:
                config_str = self._dump_data_config_to_yaml(
                    mt_path, output_dir, checkpoint_dir.name, phenotypes, variant_chunks, input_data_type
                )
                f.write(config_str + '\n')

            for phenotype_group in phenotypes:
                for phenotype in phenotype_group:
                    null_glmm = self.config.step1_null_glmm(
                        b,
                        input_bfile=input_plink_data,
                        input_phenotypes=input_phenotypes,
                        phenotype=phenotype,
                        analysis_type=SaigeAnalysisType.VARIANT,
                        covariates=covariates,
                        user_id_col=user_id_col,
                        output_dir=checkpoint_dir.name,
                    )
                    for variant_chunk in variant_chunks:
                        self.config.step2_spa(
                            b,
                            mt_path=mt_path,
                            output_dir=output_dir,
                            analysis_type=SaigeAnalysisType.VARIANT,
                            null_model=null_glmm,
                            input_data_type=input_data_type,
                            chunk=variant_chunk,
                            phenotype=phenotype,
                        )

            run_kwargs = run_kwargs or {}
            run_kwargs.pop('wait')
            b_handle = b.run(**run_kwargs, wait=False)

            with hfs.open(f'{output_dir}/batch_info.yaml', 'w') as f:
                info = yaml.safe_dump({'batch_id': b_handle.id, 'name': b.name, 'attributes': b.attributes})
                f.write(info + '\n')

            b_handle.wait()

            results_tables = []
            for phenotype_group in phenotypes:
                for phenotype in phenotype_group:
                    results_path = self.config.step2_spa.output_dir(output_dir, phenotype)
                    ht = hl.import_table(results_path, impute=True)
                    ht = ht.annotate(phenotype=phenotype.name)
                    results_tables.append(ht)

            if len(results_tables) == 1:
                results = results_tables[0]
            else:
                results = results_tables[0].union(*results_tables[1:])

            results.checkpoint(f'{output_dir}/saige_results.ht')

        return results

    def run_saige_gene(
        self,
        *,
        mt: Union[str, hl.MatrixTable],
        phenotypes: Union[List[str], Phenotypes],
        covariates: List[str],
        group_col: str,
        output_dir: str,
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        variant_chunks: Optional[VariantChunks] = None,
        run_kwargs: Optional[dict] = None,
    ):
        if checkpoint_dir is None:
            checkpoint_dir = hl.TemporaryDirectory()
        else:
            checkpoint_dir = DummyTemporaryDirectory(checkpoint_dir)

        with checkpoint_dir:
            mt, mt_path, phenotypes, variant_chunks = self._munge_inputs(
                mt=mt, checkpoint_dir=checkpoint_dir.name, phenotypes=phenotypes, variant_chunks=variant_chunks
            )

            if group_col not in mt.row:
                raise ValueError(f'could not find row annotation {group_col}')

            group = mt[group_col]
            if group.dtype != hl.tarray(hl.tstruct(group=hl.tstr, ann=hl.tstr)):
                raise ValueError(
                    f'group row annotation must have type {hl.tarray(hl.tstruct(group=hl.tstr, ann=hl.tstr))}. Found {group.dtype}'
                )

            with hfs.open(f'{output_dir}/config.yaml', 'w') as f:
                config_str = yaml.safe_dump(asdict(self.config))
                f.write(config_str + '\n')

            if b is None:
                b = hb.Batch(name=self.config.name, attributes=self.config.attributes)

            input_phenotypes, input_plink_data = self.config.prepare_inputs(b, mt, phenotypes, checkpoint_dir.name)

            user_id_col = list(mt.col_key)[0]

            if 'GP' in list(mt.entry):
                input_data_type = SaigeInputDataType.BGEN
            else:
                assert 'GT' in list(mt.entry)
                input_data_type = SaigeInputDataType.VCF

            with hfs.open(f'{output_dir}/data_config.yaml', 'w') as f:
                config_str = self._dump_data_config_to_yaml(
                    mt_path, output_dir, checkpoint_dir.name, phenotypes, variant_chunks, input_data_type
                )
                f.write(config_str + '\n')

            sparse_grm = self.config.sparse_grm(b, input_plink_data, checkpoint_dir.name)

            for phenotype_group in phenotypes:
                for phenotype in phenotype_group:
                    null_glmm = self.config.step1_null_glmm(
                        b,
                        input_bfile=input_plink_data,
                        input_phenotypes=input_phenotypes,
                        phenotype=phenotype,
                        analysis_type=SaigeAnalysisType.VARIANT,
                        covariates=covariates,
                        user_id_col=user_id_col,
                        output_dir=checkpoint_dir.name,
                        sparse_grm=sparse_grm,
                    )
                    for variant_chunk in variant_chunks:
                        self.config.step2_spa(
                            b,
                            mt_path=mt_path,
                            output_dir=output_dir,
                            analysis_type=SaigeAnalysisType.VARIANT,
                            null_model=null_glmm,
                            input_data_type=input_data_type,
                            chunk=variant_chunk,
                            phenotype=phenotype,
                            sparse_grm=sparse_grm,
                            group_col=group_col,
                        )

            run_kwargs = run_kwargs or {}
            run_kwargs.pop('wait')
            b_handle = b.run(**run_kwargs, wait=False)

            with hfs.open(f'{output_dir}/batch_info.yaml', 'w') as f:
                info = yaml.safe_dump(
                    {
                        'batch_id': b_handle.id,
                        'name': b.name,
                        'attributes': b.attributes,
                        'timestamp': str(datetime.datetime.now()),
                    }
                )
                f.write(info + '\n')

            b_handle.wait()

            # FIXME: What do saige gene results tables look like?


@app.command(help='run single variant version of SAIGE')
def run_saige(
    mt: Ann[str, Arg(help='Path to matrix table.')],
    output_dir: Ann[str, Arg(help='Path to output directory for results.')],
    phenotypes: Ann[
        List[str], Arg(help='Phenotype names to run SAIGE on. Must be valid column fields in the matrix table.')
    ],
    covariates: Ann[
        List[str], Arg(help='Covariates to use when running SAIGE. Must be valid column fields in the matrix table.')
    ],
    config: Ann[Optional[str], Opt(help='Path to SAIGE config file.')] = None,
    checkpoint_dir: Ann[Optional[str], Opt(help='Path to output directory for checkpointing files.')] = None,
    variant_chunks: Ann[
        Optional[str], Opt(help='Path to variant interval file in either bed or interval list format.')
    ] = None,
    overrides: Ann[
        Optional[List[str]],
        Opt(help='Override configuration parameters in the default configuration. Example: sparse_grm.checkpoint=true'),
    ] = None,
):
    if config is not None:
        config = SaigeConfig.from_yaml_file(config)

    if overrides:
        new_overrides = []
        for o in overrides:
            k, v = o.split('=')
            new_overrides.append((k, v))
        config = config.update(new_overrides)

    saige = SAIGE(config)

    saige.run_saige(
        mt=mt,
        phenotypes=phenotypes,
        covariates=covariates,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        variant_chunks=variant_chunks,
    )


@app.command('init', help='generate example SAIGE yaml config file')
def generate_default_config():
    config = SaigeConfig()
    print(yaml.safe_dump(asdict(config)))


if __name__ == '__main__':
    typer.run(run_saige)
