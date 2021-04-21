import json
import re
import time
import subprocess as sp


CONSEQUENCE_REGEX = re.compile(f'CSQ=[^;^\t]+')
CONSEQUENCE_HEADER_REGEX = re.compile('ID=CSQ[^>]+Description="([^"]+)')


def grouped_iterator(n, it):
    group = []
    while True:
        if len(group) == n:
            yield group
            group = []

        try:
            elt = next(it)
        except StopIteration:
            break

        group.append(elt)

    if group:
        yield group


def context():
    return '''##fileformat=VCFv4.1
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT
'''


class Variant:
    @staticmethod
    def from_vcf_line(l):
        fields = l.split('\t')
        contig = fields[0]
        pos = fields[1]
        ref = fields[3]
        alts = fields[4].split(',')
        return Variant(contig, pos, ref, alts)

    @staticmethod
    def from_string(s):
        fields = s.split(':')
        contig = fields[0]
        pos = fields[1]
        ref = fields[2]
        alts = fields[3].split(',')
        return Variant(contig, pos, ref, alts)

    def __init__(self, contig, pos, ref, alts):
        self.contig = contig
        self.position = pos
        self.ref = ref
        self.alts = alts

    def to_vcf_line(self):
        v = self.strip_star_allele()
        return f'{v.contig}\t{v.position}\t.\t{v.ref}\t{",".join(v.alts)}\t.\t.\tGT\n'

    def to_locus_alleles(self):
        return ((self.contig, self.position), self.ref + self.alts)

    def strip_star_allele(self):
        return Variant(self.contig, self.position, self.ref, [a for a in self.alts if a != '*'])

    def __str__(self):
        return f'{self.contig}:{self.position}:{self.ref}:{",".join(self.alts)}'


class Config:
    @staticmethod
    def from_file(file):
        with open(file, 'r') as f:
            config = json.loads(f.read())

        cmd = config.get('command')
        env = config.get('env')
        vep_json_schema = config.get('vep_json_schema')

        if cmd is None:
            raise ValueError("the field 'command' was not found in the config file")
        if env is None:
            raise ValueError("the field 'env' was not found in the config file")
        if vep_json_schema is None:
            raise ValueError("the field 'vep_json_schema' was not found in the config file")

        return Config(cmd, env, vep_json_schema)

    def __init__(self, cmd, env, vep_json_schema):
        self.cmd = cmd
        self.env = env
        self.vep_json_schema = vep_json_schema


def consume_header(f) -> str:
    header = ''
    line = None
    pos = 0
    while line is None or line.startswith('#'):
        pos = f.tell()
        line = f.readline()
        if line.startswith('#'):
            header += line
    f.seek(pos)
    return header


def read_config(config_file):
    return Config.from_file(config_file)


def adjust_config_cmd(config, consequence, data_dir):
    dir_found = False
    for i, c in enumerate(config.cmd):
        if c == '__OUTPUT_FORMAT_FLAG__':
            if consequence:
                config.cmd[i] = '--vcf'
            else:
                config.cmd[i] = '--json'
        elif data_dir is not None and c.startswith('--dir='):
            config.cmd[i] = f'--dir={data_dir}'
            dir_found = True
    if data_dir is not None and not dir_found:
        config.cmd.append(f'--dir={data_dir}')
    config.cmd.append(f'--dir_plugins={data_dir}Plugins')


def get_csq_header(config_file, data_dir):
    config = read_config(config_file)
    adjust_config_cmd(config, consequence=True, data_dir=data_dir)

    with sp.Popen(config.cmd, env=config.env, stdin=sp.PIPE, stdout=sp.PIPE, encoding='utf-8') as proc:
        header = context()
        v = Variant(1, 13372, 'G', ['C'])
        data = f'{header}\n{v.to_vcf_line()}'

        stdout, stderr = proc.communicate(data)
        if stderr:
            print(stderr)

        for line in stdout.split('\n'):
            line = line.rstrip()
            for match in CONSEQUENCE_HEADER_REGEX.finditer(line):
                return match.group(1)
        print('WARNING: could not get VEP CSQ header')
        return None


def run_vcf_grch38(input_file):
    import subprocess as sp
    out = sp.run('ls -l /opt/vep/.vep/', stdout=sp.PIPE, stderr=sp.STDOUT)
    print(out)

    out = sp.run('ls -l /opt/vep/Plugins/', stdout=sp.PIPE, stderr=sp.STDOUT)
    print(out)

    out = sp.run(f'''/vep --input_file {input_file} \
    --format vcf \
    --vcf \
    --everything \
    --allele_number \
    --no_stats \
    --cache \
    --offline \
    --minimal \
    --verbose \
    --assembly GRCh38 \
    --dir=/opt/vep/.vep \
    --fasta /opt/vep/.vep/homo_sapiens/95_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz \
    --plugin LoF,loftee_path:/opt/vep/Plugins/,gerp_bigwig:/opt/vep/.vep/gerp_conservation_scores.homo_sapiens.GRCh38.bw,human_ancestor_fa:/opt/vep/.vep/human_ancestor.fa.gz,conservation_file:/opt/vep/.vep/loftee.sql \
    --dir_plugins /opt/vep/Plugins/ \
    -o /tmp/test-loftee-output.vcf
''', shell=True, stderr=sp.STDOUT, stdout=sp.PIPE, env={'PERL5LIB': '/vep_data/loftee'})
    print(out.stdout)


def run_vcf_grch37(input_file):
    # Had to add loftee_path:/vep_bin/loftee in order to get the loftee plugin to be found
    # Had to add dir = /root/.vep for the cache to be found because the home dir can no longer be /vep
    import os
    os.system(f'''/vep --input_file {input_file} \
     --format vcf \
     --vcf \
     --everything \
     --allele_number \
     --no_stats \
     --cache \
     --offline \
     --minimal \
     --assembly GRCh37 \
     --dir=/root/.vep \
     --plugin LoF,loftee_path:/vep_bin/loftee,human_ancestor_fa:/root/.vep/loftee_data/human_ancestor.fa.gz,filter_position:0.05,min_intron_size:15,conservation_file:/root/.vep/loftee_data/phylocsf_gerp.sql,gerp_file:/root/.vep/loftee_data/GERP_scores.final.sorted.txt.gz
     /tmp/test-loftee-output.vcf
''')


def run(input_file, config_file, block_size, data_dir, consequence, tolerate_parse_error, part_id):
    config = read_config(config_file)
    adjust_config_cmd(config, consequence, data_dir)

    results = []

    with open(input_file, 'r') as inp:
        header = consume_header(inp)
        for block_id, block in enumerate(grouped_iterator(block_size, inp)):
            n_processed = len(block)
            start_time = time.time()

            proc_id = f'{{"part_id":{part_id},"block_id":{block_id}}}'
            variants = [Variant.from_vcf_line(l.rstrip()) for l in block]
            non_star_to_orig_variants = {str(v.strip_star_allele()): str(v) for v in variants}

            with sp.Popen(config.cmd, env=config.env, stdin=sp.PIPE, stdout=sp.PIPE,
                          stderr=sp.PIPE, encoding='utf-8') as proc:
                data = f'{header}{"".join(block)}'

                stdout, stderr = proc.communicate(data)
                if stderr:
                    print(stderr)

                for line in stdout.split('\n'):
                    line = line.rstrip()
                    print(repr(line))
                    if line != '' and not line.startswith('#'):
                        if consequence:
                            vep_v = Variant.from_vcf_line(line)
                            orig_v_str = non_star_to_orig_variants.get(str(vep_v))
                            orig_v = Variant.from_string(orig_v_str)

                            if orig_v is not None:
                                x = CONSEQUENCE_REGEX.findall(line)
                                if x:
                                    first: str = x[0]
                                    result = (orig_v, first[4:].split(','), proc_id)
                                else:
                                    print(f'WARNING: No CSQ INFO field for VEP output variant {vep_v}. VEP output is {line}')
                                    result = (orig_v, None, proc_id)
                            else:
                                raise ValueError(f'VEP output variant {vep_v} not found in original variants. VEP output is {line}')
                        else:
                            try:
                                jv = json.loads(line)
                            except json.decoder.JSONDecodeError as e:
                                msg = f'VEP failed to produce parsable JSON!\n' \
                                    f'json: {line}\n' \
                                    f'error: {e.msg}'
                                if tolerate_parse_error:
                                    print(msg)
                                    continue
                                raise Exception(msg) from e
                            else:
                                variant_string = jv.get('input')
                                if variant_string is None:
                                    raise ValueError(f'VEP generated null variant string\n'
                                                     f'json: {line}\n'
                                                     f'parsed: {jv}')
                                v = Variant.from_vcf_line(variant_string)
                                orig_v_str = non_star_to_orig_variants.get(str(v))
                                if orig_v_str is not None:
                                    orig_v = Variant.from_string(orig_v_str)
                                    result = (orig_v, line, proc_id)
                                else:
                                    raise ValueError(f'VEP output variant {vep_v} not found in original variants. VEP output is {line}')

                        results.append(result)

                if proc.returncode != 0:
                    raise ValueError(f'VEP command {" ".join(config.cmd)} failed with non-zero exit status {proc.returncode}\n'
                                     f'VEP error output:\n'
                                     f'{stderr}')

            elapsed_time = time.time() - start_time
            print(f'processed {n_processed} variants in {elapsed_time}')

    return results
