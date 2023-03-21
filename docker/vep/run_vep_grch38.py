import os
import sys

from vep import main

action = sys.argv[1]

consequence = bool(int(os.environ['VEP_CONSEQUENCE']))
tolerate_parse_error = bool(int(os.environ['VEP_TOLERATE_PARSE_ERROR']))
block_size = int(os.environ['VEP_BLOCK_SIZE'])
input_file = os.environ.get('VEP_INPUT_FILE')
output_file = os.environ['VEP_OUTPUT_FILE']
data_dir = os.environ['VEP_DATA_MOUNT']
part_id = os.environ['VEP_PART_ID']

input_file_str = f'--input_file {input_file}' if input_file else ''

vep_cmd = f'''/vep/vep \
{input_file_str} \
--format vcf {"--vcf" if consequence else "--json"} \
--everything \
--allele_number \
--no_stats \
--cache \
--offline \
--minimal \
--assembly GRCh38 \
--dir={data_dir} \
--fasta {data_dir}/homo_sapiens/95_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz \
--plugin LoF,loftee_path:/vep/ensembl-vep/Plugins/,gerp_bigwig:{data_dir}/gerp_conservation_scores.homo_sapiens.GRCh38.bw,human_ancestor_fa:{data_dir}/human_ancestor.fa.gz,conservation_file:{data_dir}/loftee.sql \
--dir_plugins /vep/ensembl-vep/Plugins/ \
-o STDOUT
'''

main(action, consequence, tolerate_parse_error, block_size, input_file, output_file, part_id, vep_cmd)
