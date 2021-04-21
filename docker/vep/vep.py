import gzip
import json
import os
import re
import shlex
import subprocess as sp
import sys
import time


CONSEQUENCE_REGEX = re.compile(r'CSQ=[^;^\t]+')
CONSEQUENCE_HEADER_REGEX = re.compile(r'ID=CSQ[^>]+Description="([^"]+)')


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


def get_csq_header(vep_cmd):
    with sp.Popen(vep_cmd, env=os.environ, stdin=sp.PIPE, stdout=sp.PIPE, encoding='utf-8') as proc:
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


def run_vep(vep_cmd, input_file, block_size, consequence, tolerate_parse_error, part_id, env):
    results = []

    with open(input_file, 'r') as inp:
        header = consume_header(inp)
        for block_id, block in enumerate(grouped_iterator(block_size, inp)):
            n_processed = len(block)
            start_time = time.time()

            proc_id = f'{{"part_id":{part_id},"block_id":{block_id}}}'
            variants = [Variant.from_vcf_line(l.rstrip()) for l in block]
            non_star_to_orig_variants = {str(v.strip_star_allele()): str(v) for v in variants}

            with sp.Popen(vep_cmd, env=env, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf-8') as proc:
                data = f'{header}{"".join(block)}'

                stdout, stderr = proc.communicate(data)
                if stderr:
                    print(stderr)

                for line in stdout.split('\n'):
                    line = line.rstrip()
                    if line != '' and not line.startswith('#'):
                        if consequence:
                            vep_v = Variant.from_vcf_line(line)
                            orig_v_str = non_star_to_orig_variants.get(str(vep_v))
                            orig_v = Variant.from_string(orig_v_str)

                            if orig_v is not None:
                                x = CONSEQUENCE_REGEX.findall(line)
                                if x:
                                    first: str = x[0]
                                    result = (orig_v, json.dumps(first[4:].split(',')), proc_id)
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
                    raise ValueError(f'VEP command {vep_cmd} failed with non-zero exit status {proc.returncode}\n'
                                     f'VEP error output:\n'
                                     f'{stderr}')

            elapsed_time = time.time() - start_time
            print(f'processed {n_processed} variants in {elapsed_time}')

    return results


def main(action: str,
         consequence: bool,
         tolerate_parse_error: bool,
         block_size: int,
         input_file: str,
         output_file: str,
         part_id: str,
         vep_cmd: str):
    vep_cmd = shlex.split(vep_cmd)

    if action == 'csq_header':
        csq_header = get_csq_header(vep_cmd)
        with open(output_file, 'w') as out:
            out.write(f'{csq_header}\n')
    else:
        assert action == 'vep'
        results = run_vep(vep_cmd, input_file, block_size, consequence, tolerate_parse_error, part_id, os.environ)
        with gzip.open(output_file, 'wt') as out:
            out.write(f'variant\tvep\tvep_proc_id\n')
            for v, a, proc_id in results:
                out.write(f'{v}\t{a}\t{proc_id}\n')


if __name__ == '__main__':
    action = sys.argv[1]

    consequence = bool(int(os.environ['VEP_CONSEQUENCE']))
    tolerate_parse_error = bool(int(os.environ['VEP_TOLERATE_PARSE_ERROR']))
    block_size = int(os.environ['VEP_BLOCK_SIZE'])
    input_file = os.environ.get('VEP_INPUT_FILE')
    output_file = os.environ['VEP_OUTPUT_FILE']
    data_dir = os.environ['VEP_DATA_MOUNT']
    part_id = os.environ['VEP_PART_ID']
    vep_cmd = os.environ['VEP_COMMAND']

    main(action, consequence, tolerate_parse_error, block_size, input_file, output_file, part_id, vep_cmd)
