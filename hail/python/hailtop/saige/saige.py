from collections import namedtuple
from typing import Dict, Optional

import hail as hl
from hail.backend.service_backend import ServiceBackend
import hailtop.batch as hb
import hailtop.batch_client as bc
from hailtop.utils import secret_alnum_string

from .config import SaigeConfig
from .constants import SaigeAnalysisType, SaigePhenotype
from .io import PlinkInputFile
from .null_model import fit_null_glmm
from .sparse_grm import create_sparse_grm
from .utils import rectify_attributes, rectify_name


SaigePlinkInputResources = namedtuple('SaigePlinkInputResources', ['input_bfile', 'phenotypes', 'covariates'])


def get_saige_plink_inputs(b: hb.Batch,
                           input_bfile: str,
                           pheno_file: str,
                           covariates_file: str) -> SaigePlinkInputResources:
    input_bfile_resource = PlinkInputFile.from_root_path(input_bfile).to_resource(b)
    phenotypes = b.read_input(pheno_file)
    covariates = b.read_input(covariates_file)
    return SaigePlinkInputResources(input_bfile=input_bfile_resource, phenotypes=phenotypes, covariates=covariates)


def get_saige_batch(name: Optional[str], attributes: Optional[Dict[str, str]]):
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)

    token = secret_alnum_string(16)
    if backend._batch is None:
        backend._batch = backend.async_bc.create_batch(
            name=name, token=token, attributes=attributes
        )

    return bc.client.Batch(backend._batch)


def run_saige(config: SaigeConfig,
              input_plink_bfile: str,
              pheno_file: str,
              covariates_file: str,
              output_root: str,
              trait_type: SaigePhenotype,
              analysis_type: SaigeAnalysisType,
              *,
              name: Optional[str] = None,
              attributes: Optional[Dict[str, str]] = None):
    hl.init()
    name = rectify_name(config.name, name)
    attributes = rectify_attributes(config.attributes, attributes)

    b = get_saige_batch(name, attributes)
    inputs = get_saige_plink_inputs(b, input_plink_bfile, pheno_file, covariates_file)

    sparse_grm = create_sparse_grm(b,
                                   config.sparse_grm_config,
                                   input_bfile=inputs.input_bfile,
                                   )

    null_model = fit_null_glmm(b,
                               config.null_model_config,
                               input_bfile=inputs.input_bfile,
                               phenotypes=inputs.phenotypes,
                               covariates=inputs.covariates,
                               output_root=output_root,
                               trait_type=trait_type,
                               analysis_type=analysis_type)







    # create null model
    #