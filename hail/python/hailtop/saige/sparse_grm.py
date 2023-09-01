from typing import Dict, Optional

import hailtop.batch as hb

from .config import SaigeSparseGrmConfig
from .io import SaigeSparseGRMResource
from .utils import rectify_attributes, rectify_name


def create_sparse_grm(b: hb.Batch,
                      config: SaigeSparseGrmConfig,
                      *,
                      input_bfile: hb.ResourceGroup,
                      output_path: str,
                      name: Optional[str] = None,
                      attributes: Optional[Dict[str, str]] = None
                      ):
    name = rectify_name(config.name, name)
    attributes = rectify_attributes(config.attributes, attributes)

    create_sparse_grm_task = b.new_job(name=name, attributes=attributes)

    (create_sparse_grm_task
     .cpu(config.n_threads)
     .storage(config.storage)
     .image(config.docker_image)
     )

    relatedness_cutoff = config.relatedness_cutoff
    num_markers = config.num_markers

    mtx_identifier = f'_relatednessCutoff_{relatedness_cutoff}_{num_markers}_randomMarkersUsed.sparseGRM.mtx'

    sparse_grm_output = SaigeSparseGRMResource(create_sparse_grm_task, mtx_identifier=mtx_identifier)

    command = f'''
Rscript /usr/local/bin/createSparseGRM.R \
    --plinkFile={input_bfile} \
    --nThreads={config.n_threads} \
    --outputPrefix={sparse_grm_output.mtx_identifier} \
    --numRandomMarkerforSparseKin={config.num_markers} \
    --relatednessCutoff={relatedness_cutoff}
'''

    create_sparse_grm_task.command(command)

    b.write_output(sparse_grm_output.resource, output_path)

    return sparse_grm_output
