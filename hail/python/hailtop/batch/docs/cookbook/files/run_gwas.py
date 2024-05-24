import argparse

import hail as hl


def run_gwas(vcf_file, phenotypes_file, output_file):
    table = hl.import_table(phenotypes_file, impute=True).key_by('Sample')

    hl.import_vcf(vcf_file).write('tmp.mt')
    mt = hl.read_matrix_table('tmp.mt')

    mt = mt.annotate_cols(pheno=table[mt.s])
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols((mt.sample_qc.dp_stats.mean >= 4) & (mt.sample_qc.call_rate >= 0.97))
    ab = mt.AD[1] / hl.sum(mt.AD)
    filter_condition_ab = (
        (mt.GT.is_hom_ref() & (ab <= 0.1))
        | (mt.GT.is_het() & (ab >= 0.25) & (ab <= 0.75))
        | (mt.GT.is_hom_var() & (ab >= 0.9))
    )
    mt = mt.filter_entries(filter_condition_ab)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.AF[1] > 0.01)

    eigenvalues, pcs, _ = hl.hwe_normalized_pca(mt.GT)

    mt = mt.annotate_cols(scores=pcs[mt.s].scores)

    gwas = hl.linear_regression_rows(
        y=mt.pheno.CaffeineConsumption,
        x=mt.GT.n_alt_alleles(),
        covariates=[1.0, mt.pheno.isFemale, mt.scores[0], mt.scores[1], mt.scores[2]],
    )

    gwas = gwas.select(SNP=hl.variant_str(gwas.locus, gwas.alleles), P=gwas.p_value)
    gwas = gwas.key_by(gwas.SNP)
    gwas = gwas.select(gwas.P)
    gwas.export(f'{output_file}.assoc', header=True)

    hl.export_plink(mt, output_file, fam_id=mt.s, ind_id=mt.s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vcf', required=True)
    parser.add_argument('--phenotypes', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--cores', required=False)
    args = parser.parse_args()

    if args.cores:
        hl.init(master=f'local[{args.cores}]')

    run_gwas(args.vcf, args.phenotypes, args.output_file)
