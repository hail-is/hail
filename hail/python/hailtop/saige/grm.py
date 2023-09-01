# import copy
# from typing import Dict, Optional
#
# import hailtop.batch as hb
#
# from .config import SaigeSparseGrmConfig
#
#
# def create_sparse_grm(b: hb.Batch,
#                       config: SaigeSparseGrmConfig,
#                       *,
#                       input_plink_bfile: hb.ResourceGroup,
#                       name: Optional[str] = None,
#                       attributes: Optional[Dict[str, str]] = None):
#     name = name or config.name
#
#     if attributes is None:
#         attributes = {}
#     config_attributes = copy.deepcopy(config.attributes)
#     attributes.update(config_attributes)
#
#     grm_job = b.new_job(name=name)
#
#     (grm_job
#      .cpu(config.n_threads)
#      .storage(config.storage)
#      .image(config.docker_image)
#      )
#
#     grm_job.declare_resource_group(
#         sparse_grm={
#             ext: f'{{root}}{ext}' for ext in
#             (f'_relatednessCutoff_{config.relatedness_cutoff}_{config.num_markers}_randomMarkersUsed.sparseGRM.mtx',
#              f'_relatednessCutoff_{config.relatedness_cutoff}_{config.num_markers}_randomMarkersUsed.sparseGRM.mtx.sampleIDs.txt')
#         }
#     )
#
#     command = f'''
# Rscript /usr/local/bin/createSparseGRM.R \
#     --plinkFile={in_bfile} \
#     --nThreads={config.n_threads} \
#     --outputPrefix={grm_job.sparse_grm} \
#     --numRandomMarkerforSparseKin={config.num_markers} \
#     --relatednessCutoff={config.relatedness_cutoff}
# '''
#
#     grm_job.command(command)
#     b.write_output(grm_job.sparse_grm, config.output_path)
#
#     return grm_job.sparse_grm
