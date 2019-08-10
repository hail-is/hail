"""A high level script for running the hail gVCF combiner/joint caller"""
import argparse
import time
import sys
import uuid

import hail as hl

from hail.experimental import vcf_combiner as comb

MAX_COMBINER_LENGTH = 100
DEFAULT_REF = 'GRCh38'

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def run_combiner(sample_list, intervals, out_path, tmp_path, summary_path=None, overwrite=False):
    import gc
    # make the temp path a directory, no matter what
    tmp_path += f'/combiner-temporary/{uuid.uuid4()}/'
    vcfs = [comb.transform_one(vcf)
            for vcf in hl.import_vcfs(sample_list, intervals, array_elements_required=False)]
    combined = [comb.combine_gvcfs(mts) for mts in chunks(vcfs, MAX_COMBINER_LENGTH)]
    if len(combined) == 1:
        combined[0].write(out_path, overwrite=overwrite)
    else:
        hl.utils.java.info(f'Writing combiner temporary files to: {tmp_path}')
        i = 0
        while len(combined) > 1:
            pad = len(str(len(combined)))
            hl.experimental.write_matrix_tables(combined, tmp_path + f'{i}/', overwrite=True)
            paths = [tmp_path + f'{i}/' + str(n).zfill(pad) + '.mt' for n in range(len(combined))]
            i += 1
            wmts = [hl.read_matrix_table(path) for path in paths]
            combined = [comb.combine_gvcfs(mts) for mts in chunks(wmts, MAX_COMBINER_LENGTH)]
            gc.collect()  # need to try to free memory on the master
        combined[0].write(out_path, overwrite=overwrite)
    if summary_path is not None:
        mt = hl.read_matrix_table(out_path)
        comb.summarize(mt).rows().write(summary_path, overwrite=overwrite)


def build_sample_list(sample_map_file, sample_list_file=None):
    if sample_list_file is None:
        with open(sample_map_file) as smap:
            return [l.strip().split('\t')[1] for l in smap]
    # else
    with open(sample_map_file) as smap:
        sample_map = dict()
        for l in smap:
            k, v = l.strip().split('\t')
            sample_map[k] = v
    with open(sample_list_file) as slist:
        sample_set = {l.strip() for l in slist}
        sample_lst = list(sample_set)
        sample_lst.sort()

    samples = []
    missing = []
    for sample in sample_lst:
        try:
            samples.append(sample_map[sample])
        except KeyError:
            missing.append(sample)
    print(f'No gVCF path for samples {", ".join(missing)}', file=sys.stderr)
    return samples

def main():
    parser = argparse.ArgumentParser(description="Driver for hail's gVCF combiner")
    parser.add_argument('--sample-map', help='path to the sample map (must be filesystem local)',
                        required=True)
    parser.add_argument('--sample-file', help='path to a file containing a line separated list'
                                              'of samples to combine (must be filesystem local)')
    parser.add_argument('--tmp-path', help='path to folder for temp output (can be a cloud bucket)',
                        default='/tmp')
    parser.add_argument('--out-file', '-o', help='path to final combiner output', required=True)
    parser.add_argument('--summarize', help='if defined, run summarize, placing the rows table '
                                            'of the output at the argument value')
    parser.add_argument('--json', help='json to use for the import of the gVCFs'
                                       '(must be filesystem local)', required=True)
    args = parser.parse_args()
    samples = build_sample_list(args.sample_map, args.sample_file)
    with open(args.json) as j:
        ty = hl.tarray(hl.tinterval(hl.tstruct(locus=hl.tlocus(reference_genome='GRCh38'))))
        intervals = ty._from_json(j.read())
    hl.init(default_reference=DEFAULT_REF,
            log='/hail-joint-caller-' + time.strftime('%Y%m%d-%H%M') + '.log')
    run_combiner(samples, intervals, args.out_file, args.tmp_path, summary_path=args.summarize,
                 overwrite=True)


if __name__ == '__main__':
    main()
