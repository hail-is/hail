"""A high level script for running the hail gVCF combiner/joint caller"""
import argparse
import time
import uuid

import hail as hl

from hail.experimental import vcf_combiner as comb

DEFAULT_REF = 'GRCh38'
MAX_MULTI_WRITE_NUMBER = 50
MAX_COMBINE_NUMBER = 100

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def h(paths, sample_names, tmp_path, json, header, out_path, i):
    vcfs = [comb.transform_one(vcf)
            for vcf in hl.import_vcfs(paths, json, array_elements_required=False,
                                      _external_header=header,
                                      _external_sample_ids=sample_names)]
    combined = [comb.combine_gvcfs(mts) for mts in chunks(vcfs, MAX_COMBINE_NUMBER)]
    if len(paths) <= MAX_COMBINE_NUMBER:  # only 1 item, just write it
        combined[0].write(out_path, overwrite=True)
        return []
    pad = len(str(len(combined)))
    hl.experimental.write_matrix_tables(combined, tmp_path + f'{i}/', overwrite=True)
    return [tmp_path + f'{i}/' + str(n).zfill(pad) + '.mt' for n in range(len(combined))]

def stage_one(paths, sample_names, tmp_path, json, header, out_path):
    assert len(paths) == len(sample_names)
    tmp_path += f'{uuid.uuid4()}/'
    out_paths = []
    i = 0
    size = MAX_MULTI_WRITE_NUMBER * MAX_COMBINE_NUMBER
    for pos in range(0, len(paths), size):
        i += 1
        tmp = h(paths[pos:pos + size], sample_names[pos:pos + size], tmp_path, json, header,
                out_path, i)
        if not tmp:
            return tmp
        out_paths.extend(tmp)
    return out_paths

def run_combiner(samples, intervals, out_file, tmp_path, header, overwrite=True):
    tmp_path += f'/combiner-temporary/{uuid.uuid4()}/'
    sample_names, paths = [list(x) for x in zip(*samples)]
    sample_names = [[n] for n in sample_names]
    assert len(paths) == len(samples)
    out_paths = stage_one(paths, sample_names, tmp_path, intervals, header, out_file)
    if not out_paths:
        return
    tmp_path += f'{uuid.uuid4()}/'
    mts = [hl.read_matrix_table(path) for path in out_paths]
    combined_mts = [comb.combine_gvcfs(mt) for mt in chunks(mts, MAX_COMBINE_NUMBER)]
    i = 0
    while len(combined_mts) > 1:
        tmp = tmp_path + f'{i}/'
        pad = len(str(len(combined_mts)))
        hl.experimental.write_matrix_tables(combined_mts, tmp, overwrite=True)
        paths = [tmp + str(n).zfill(pad) + '.mt' for n in range(len(combined_mts))]
        mts = [hl.read_matrix_table(path) for path in paths]
        combined_mts = [comb.combine_gvcfs(mts) for mt in chunks(mts, MAX_COMBINE_NUMBER)]
    combined_mts[0].write(out_file, overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(description="Driver for hail's gVCF combiner")
    parser.add_argument('--sample-map',
                        help='path to the sample map (must be filesystem local),'
                             ' sorted by schemata used in sample',
                        required=True)
    parser.add_argument('--tmp-path', help='path to folder for temp output (can be a cloud bucket)',
                        default='/tmp')
    parser.add_argument('--out-file', '-o', help='path to final combiner output', required=True)
    parser.add_argument('--json', help='json to use for the import of the gVCFs'
                                       '(must be filesystem local)', required=True)
    parser.add_argument('--header', help='external header, must be cloud based', required=True)
    args = parser.parse_args()
    hl.init(default_reference=DEFAULT_REF,
            log='/hail-joint-caller-' + time.strftime('%Y%m%d-%H%M') + '.log')
    with open(args.json) as j:
        ty = hl.tarray(hl.tinterval(hl.tstruct(locus=hl.tlocus(reference_genome='GRCh38'))))
        intervals = ty._from_json(j.read())
    with open(args.sample_map) as m:
        samples = [l.strip().split('\t') for l in m]
    run_combiner(samples, intervals, args.out_file, args.tmp_path, args.header, overwrite=True)


if __name__ == '__main__':
    main()
