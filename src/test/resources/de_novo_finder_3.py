#!/usr/bin/env python
'''
Find potential de novo variants in a given VCF.

de_novo_finder_3.py is a major update to the original caller. The main purpose is to
identify events that appear to be de novo in a specified VCF file that contains
sequence information from trios. Due to the nature of this variation, parents are required
to be homozygous reference, while children should usually be heterozygous for the
variant. Confidence in the calls is established with quality filters:

1) The variant must pass all of the filters applied by the variant caller
    To accept TruthSensitivityTranche variants, use the -q flag.
    ** Removed in 3.93 **
2) The PL (normalized, Phred-scaled likelihoods for AA, AB, and BB genotypes where
A = ref and B = alt) score of the child is required to be >T : 0 : >0 for a given
threshold, T.
    T is set at a default of 20. It can be adjusted with the -t flag.
3) The allelic balance (# alternative reads/total reads) of the child is required
to be at least 20%. The allelic balance in the parents should be less than or
equal to 5%.
    These numbers can be adjusted with the -c and -p flags, respectively.
4) The depth in the child is required to be greater than a tenth of the sum of the
depths in both parents.
    The fraction of the sum of depths can be adjusted with the -d flag.

This script processes both single nucleotide variants (SNVs) and small insertions and
deletions (indels). To skip indels, use the -i flag. ** Removed in 3.81 **

Lines in the VCF that have multiple alternative alleles are processed only if all
alleles are single bases.

Instead of requiring a hard PL threshold in the parents, we have defined a relative
probability of an event being truly de novo versus the probability that it was a missed
heterozygote call in one of the two parents (the most likely error mode).
p_dn = P(true de novo | data) / (P(true de novo | data) + P(missed het in parent | data))

where P(true de novo | data) = P(data | true de novo) * P(true de novo)
P(data | true de novo) = Pdad_ref * Pmom_ref * Pchild_het
P(true de novo) = 1/30 Mb
and P(missed het in parent | data) = P(data | at least one parent is het) * P(one parent het)
P(data | at least one parent is het) = (Pdad_ref*Pmom_het + Pdad_het*Pmom_ref) * Pchild_het
P(one parent het) = 1 - (1-f)^4
where f is the maximum of the frequency of the variant in the VCF or ESP

The minimum p_dn considered is 0.05, but can be adjusted with the -m flag.

The potential de novo variants are then split by SNVs and indels into HIGH, MEDIUM,
and LOW validation likelihood by the following criteria.
HIGH_SNV:
p_dn > 0.99 and child_AD > 0.3 and dp_ratio (kid depth/combined parental depth) > 0.2
or
p_dn > 0.99 and child_AD > 0.3 and allele count (AC) == 1

MEDIUM_SNV:
p_dn > 0.5 and child_AD > 0.3
or
p_dn > 0.5 and child_AD  > 0.2 and AC == 1

LOW_SNV:
p_dn > 0.05 and child_AD > 0.2


HIGH_indel:
p_dn > 0.99 and child_AD > 0.3 and dp_ratio > 0.2
or
p_dn > 0.99 and child_AD > 0.3 and AC == 1

MEDIUM_indel:
p_dn > 0.5 and child_AD > 0.3
or
p_dn > 0.5 and child_AD > 0.2 and AC ==1

LOW_indel:
p_dn > 0.05 and child_AD > 0.2


If SnpEff annotations have been included in the annotation line of the VCF, the -a
flag can be used to extract and print the gene name and mutation category. The same is
true for VEP annotations using the -v flag.

The updates to this caller require changes in the input. A PED file is required to
establish the family relations. PED format has 6 columns: family ID, child, dad, mom,
sex of the child, affected status of the child. The ESP counts file is required to
determine the population frequency of an event.

Current ESP counts file: all_ESP_counts_5.28.13.txt

Output contains many more columns than earlier versions of the script so downstream
scripts may need to be adjusted.


3.4: different flags
3.5: fixed depth bug and now prints DP_ratio, child sex and affected status
3.52: modified to remove "chr" if it is in the "chr" column and skip lines
that have no PL information
3.6: Changed the INCORRECT MEDIUM_indel variant AC requirement. It used
to be AC >= 5, but it should really be AC <= 5. (This may been seen in
some versions of 3.52.) In addtion, 3.6 skips lines with no AD information
and keeps track of the number.
3.7: Adjusted validation likelihood filters so that MEDIUM_SNV that meet
the following criteria are moved to HIGH_SNV:
    AC < 10
    child AD > 0.3
    child depth > 10
3.71: Added ability to handle .gz VCFs
3.72: Haplotype caller for 0/0 individuals will list an AD of "." which breaks the
script. I am now assuming that there are 0 alternative reads in these cases.
3.73: Haplotype caller now for the hemizygous variants
3.74: Fixed the labels reading line so that it strips off the newline character
3.75: Added the ability to handle VEP annotations (curtesy of Jack Kosmicki)
3.8: The VEP annotation section was completely wrong. Rewrote with code from Konrad
Karczewski loftee_utils.py (https://github.com/konradjk/loftee/blob/master/src/loftee_utils.py)
3.9: DROPPED THE -i FLAG. The script does not throw out multiallelic lines where a SNV
and indel are present. It, however, still does not handle multialleleic lines with more
than two alternative alleles.
3.91: Slight update to VEP annotation list
3.92: Adding a quick fix to avoid situations where SnpEff where EFFECT (such as
'INTRON') is missing
3.93: Removed the -q flag that allowed individuals to look at Truth Sensitivity
Tranche variants
3.94: Had never incorporated that if the greatest frequency (f) == 0, then it
becomes 100/30Mbp
3.95: Fixed a bug that led to the first variant being dropped from the VCF

'''

__version__ = 3.95
__author__ = 'Kaitlin E. Samocha <ksamocha@fas.harvard.edu>'
__date__ = 'March 10th, 2016'

import argparse
import os.path
import sys
import re
import time
import gzip

# Note that this list of VEP annotations is current as of v77 with 2 included for backwards compatibility (VEP <= 75)
# From Konrad Karczewski
# Slight updates from Jack Kosmicki
csq_order = ["transcript_ablation",
             "splice_donor_variant",
             "splice_acceptor_variant",
             "stop_gained",
             "frameshift_variant",
             "stop_lost",
             "start_lost",
             "initiator_codon_variant",  # deprecated
             "transcript_amplification",
             "inframe_insertion",
             "inframe_deletion",
             "missense_variant",
             "protein_altering_variant",
             "splice_region_variant",
             "incomplete_terminal_codon_variant",
             "stop_retained_variant",
             "synonymous_variant",
             "coding_sequence_variant",
             "mature_miRNA_variant",
             "5_prime_UTR_variant",
             "3_prime_UTR_variant",
             "non_coding_transcript_exon_variant",
             "non_coding_exon_variant",  # deprecated
             "intron_variant",
             "NMD_transcript_variant",
             "non_coding_transcript_variant",
             "nc_transcript_variant",  # deprecated
             "upstream_gene_variant",
             "downstream_gene_variant",
             "TFBS_ablation",
             "TFBS_amplification",
             "TF_binding_site_variant",
             "regulatory_region_ablation",
             "regulatory_region_amplification",
             "feature_elongation",
             "regulatory_region_variant",
             "feature_truncation",
             "intergenic_variant",
             ""]
csq_order_dict = dict(zip(csq_order, range(len(csq_order))))
rev_csq_order_dict = dict(zip(range(len(csq_order)), csq_order))


def trimfamily(Fam, labels):
    "Trim family dictionary to leave only families that are in the VCF"
    Fam2 = {}
    for child, (dad, mom, gender, aff_status) in Fam.items():
        if child not in labels:
            sys.stderr.write("Could not find child: {0}\n".format(child))
            continue

        if dad not in labels:
            sys.stderr.write("Could not find dad: {0}\n".format(dad))
            continue

        if mom not in labels:
            sys.stderr.write("Could not find mom: {0}\n".format(mom))
            continue

        if gender in ('Male', 'male', 'M', 'm', '1'):
            gender = 1
        elif gender in ('Female', 'female', 'F', 'f', '2'):
            gender = 2
        else:
            gender = 0  # consider it as missing

        child = labels.index(child)
        dad = labels.index(dad)
        mom = labels.index(mom)
        Fam2[child] = (dad, mom, gender, aff_status)

    # Now to make the arrays for easier look up
    am_kid = ['N' for i in range(9, len(labels))]
    who_dad = ['N' for i in range(9, len(labels))]
    who_mom = ['N' for i in range(9, len(labels))]
    for idx in range(9, len(labels)):
        if idx in Fam2.keys():
            am_kid[idx - 9] = 'Y'
            (dad_pos, mom_pos, gender, aff_status) = Fam2[idx]
            who_dad[idx - 9] = dad_pos
            who_mom[idx - 9] = mom_pos

    return (Fam2, am_kid, who_dad, who_mom)


def split_Fam(Fam_dict, labels):
    "Split the family dictionary into a female vs male children"

    fem_Fam = {}
    male_Fam = {}
    female_kid = ['N' for i in range(9, len(labels))]
    male_kid = ['N' for i in range(9, len(labels))]

    for family, fam_info in Fam_dict.items():
        if fam_info[2] == 1:  # male
            male_Fam[family] = fam_info
        elif fam_info[2] == 2:  # female
            fem_Fam[family] = fam_info
        else:
            continue

    for idx in range(9, len(labels)):
        if idx in fem_Fam.keys():
            female_kid[idx - 9] = 'Y'
        elif idx in male_Fam.keys():
            male_kid[idx - 9] = 'Y'
        else:
            continue

    return (fem_Fam, female_kid, male_Fam, male_kid)


def process_line(line, args):
    '''Processes the variant line and runs quality checks on the variant

    May not need to be its own function given the removal of -q in 3.93
    '''
    (ref_allele, alt_allele, qual, filter_pos) = range(3, 7)
    passedcheck = True

    if line[filter_pos] != 'PASS':
        passedcheck = False

    return passedcheck


def is_child(column, Fam):
    "Checks if the het entry is a child"
    if column in Fam.keys():
        return (Fam[column][0], Fam[column][1])  # Should return the parents

    return None


def child_cuts(record, PL_pos, AD_pos, args):
    "Apply the PL and AD filters"
    # To skip lines that have no PL information
    try:
        PL = record[PL_pos].split(",")
    except IndexError:
        return None

    if int(PL[0]) <= args.thresh:
        return None
    if PL[1] != "0":
        return None

    if record[AD_pos] == '.':
        sys.stderr.write('Child had AD of ".": {0}\n'.format(record))
        return None

    AD = record[AD_pos].split(',')  # should be for hets only. not modifying
    if ((AD[0] == '0') and (AD[1] == '0')):
        return None

    ratio = float(AD[1]) / (float(AD[0]) + float(AD[1]))
    if ratio <= args.minchildAB:
        return None

    return (PL, ratio)


def DPcheck(DPlist, args):  # revamped
    "Check that the child's depth is appropriate given the parents' depths"
    percent_depthratio = args.depthratio / 100.0
    dp_ratio = float(DPlist[0]) / (float(DPlist[1]) + float(DPlist[2]))
    if dp_ratio >= percent_depthratio:
        return dp_ratio
    else:
        return None


def snpeff_annotate(var_annotation):
    "Take variant annotation and extract gene name and mutation type"
    # Set for SnpEff annotation
    var_annotation = var_annotation.split(';')

    gene_name = "."
    functional_class = "."
    effect = "."

    for entry in var_annotation:
        if re.search('GENE_NAME', entry):
            gene_name = entry.split('=')[1]
        elif re.search('FUNCTIONAL_CLASS', entry):
            functional_class = entry.split('=')[1]
        elif re.search('EFFECT', entry):
            effect = entry.split('=')[1]
        else:
            continue

    if functional_class in ('NONSENSE', 'MISSENSE', 'SILENT'):
        return (gene_name, functional_class)
    elif (functional_class == 'NONE'):
        return (gene_name, effect)
    else:
        return (gene_name, 'NA')


def VEP_annotate(var_annotation, vep_field_names, alt_allele):
    '''Take variant annotation and extract gene name and mutation type
    Set for VEP annotation, which has all the crazy pipes
    Based on code from Konrad Karczewski
    '''
    gene_name = "."
    functional_class = "."

    info_field = dict([(x.split('=', 1)) if '=' in x else (x, x) for x in re.split(';(?=\w)', var_annotation)])
    if 'CSQ' not in info_field:
        return (gene_name, functional_class)

    # array with dictionaries containing the information
    annotations = [dict(zip(vep_field_names, x.split('|'))) for x in info_field['CSQ'].split(',') if
                   len(vep_field_names) == len(x.split('|'))]

    # loop through and choose the canonical annotation
    # check that alternative allele matches
    for entry in annotations:
        if entry['Allele'] != alt_allele:
            continue
        if entry['CANONICAL'] != 'YES':
            continue
        gene_name = entry['SYMBOL']
        entry['major_consequence'] = worst_csq_from_csq(entry['Consequence'])
        functional_class = entry['major_consequence']

    # If there is no canonical transcript, return worst consequence
    # Code taken from loftee_utils.py by Konrad Karczewski
    if gene_name == '.':
        worst_annotation = worst_csq_with_vep(annotations)
        if worst_annotation != None:
            gene_name = worst_annotation['SYMBOL']
            functional_class = worst_annotation['major_consequence']

    return (gene_name, functional_class)


def worst_csq_with_vep(annotation_list):
    """
    Takes list of VEP annotations [{'Consequence': 'frameshift', Feature: 'ENST'}, ...]
    Returns most severe annotation (as full VEP annotation [{'Consequence': 'frameshift', Feature: 'ENST'}])
    Also tacks on worst consequence for that annotation (i.e. worst_csq_from_csq)
    :param annotation_list:
    :return worst_annotation:
    """
    if len(annotation_list) == 0: return None
    worst = annotation_list[0]
    for annotation in annotation_list:
        if compare_two_consequences(annotation['Consequence'], worst['Consequence']) < 0:
            worst = annotation
        elif compare_two_consequences(annotation['Consequence'], worst['Consequence']) == 0 and annotation[
            'CANONICAL'] == 'YES':
            worst = annotation
    worst['major_consequence'] = worst_csq_from_csq(worst['Consequence'])
    return worst


def compare_two_consequences(csq1, csq2):
    'From Konrad Karczewski'
    if csq_order_dict[worst_csq_from_csq(csq1)] < csq_order_dict[worst_csq_from_csq(csq2)]:
        return -1
    elif csq_order_dict[worst_csq_from_csq(csq1)] == csq_order_dict[worst_csq_from_csq(csq2)]:
        return 0
    return 1


def worst_csq_from_csq(csq):
    """
    Input possibly &-filled csq string (e.g. 'non_coding_exon_variant&nc_transcript_variant')
    Return the worst annotation (In this case, 'non_coding_exon_variant')
    :param consequence:
    :return most_severe_consequence:

    From Konrad Karczewski
    """
    return rev_csq_order_dict[worst_csq_index(csq.split('&'))]


def worst_csq_index(csq_list):
    """
    Input list of consequences (e.g. ['frameshift_variant', 'missense_variant'])
    Return index of the worst annotation (In this case, index of 'frameshift_variant', so 4)
    Works well with csqs = 'non_coding_exon_variant&nc_transcript_variant' by worst_csq_index(csqs.split('&'))
    :param annnotation:
    :return most_severe_consequence_index:

    From Konrad Karczewski
    """
    return min([csq_order_dict[ann] for ann in csq_list])


def parent_AD_cuts(dad_AD_info, mom_AD_info, args):
    "Apply the AD filter to the parental data"
    if dad_AD_info == '.':
        dad_AD_ratio = 0.0
    else:
        dad_AD = dad_AD_info.split(',')
        if ((dad_AD[0] == '0') and (dad_AD[1] == '0')):
            return None
        dad_AD_ratio = float(dad_AD[1]) / (float(dad_AD[0]) + float(dad_AD[1]))

    if mom_AD_info == '.':
        mom_AD_ratio = 0.0
    else:
        mom_AD = mom_AD_info.split(',')
        if ((mom_AD[0] == '0') and (mom_AD[1] == '0')):
            return None
        mom_AD_ratio = float(mom_AD[1]) / (float(mom_AD[0]) + float(mom_AD[1]))

    if ((args.maxparentAB <= dad_AD_ratio) or
            (args.maxparentAB <= mom_AD_ratio)):
        return None

    return (dad_AD_ratio, mom_AD_ratio)


def load_esp_counts(esp_file, chrom):
    "Open the ESP counts file and save variants for a given chromosome"
    esp_counts = {}
    (count_ea, numchr_ea, af_ea, count_aa, numchr_aa, af_aa) = range(9, 15)

    with open(esp_file, 'r') as esp_data:
        for line in esp_data:
            line = line.split()

            if line[0] != chrom:
                continue

            # key format -- chr:pos:ref:alt
            chr_pos_change = '{0}:{1}:{2}:{3}'.format(line[0], line[1],
                                                      line[2], line[3])
            # value format -- frequency
            allele_count = float(line[count_ea]) + float(line[count_aa])
            chr_count = float(line[numchr_ea]) + float(line[numchr_aa])
            allele_freq = allele_count / chr_count

            esp_counts[chr_pos_change] = allele_freq

    return esp_counts


def determine_validation_likelihood(ref, alt, child_AD, annotation, p_dn, dp_ratio, child_dp):
    '''Determine the likelihood of a de novo variant validating (HIGH, MEDIUM,
    LOW) split by SNVs and indels'''
    qual_flag = 'None'
    variant_AC = annotation.split(';')[0].split('=')[1]

    # Indels
    if (ref not in ('A', 'C', 'G', 'T')) or (alt not in ('A', 'C', 'G', 'T')):
        if (p_dn > 0.99) and (child_AD > 0.3) and (variant_AC == '1'):
            qual_flag = 'HIGH_indel'
        elif (p_dn > 0.5) and (child_AD > 0.3) and (float(variant_AC) <= 5):
            qual_flag = 'MEDIUM_indel'
        elif (p_dn > 0.05) and (child_AD > 0.2):
            qual_flag = 'LOW_indel'

    # SNVs
    else:
        if (p_dn > 0.99) and (child_AD > 0.3) and (dp_ratio > 0.2):
            qual_flag = 'HIGH_SNV'
        elif (p_dn > 0.99) and (child_AD > 0.3) and (variant_AC == '1'):
            qual_flag = 'HIGH_SNV'

        # Added to move some MEDIUM variants into HIGH
        elif (p_dn > 0.5) and (child_AD >= 0.3) and (float(variant_AC) < 10) and (float(child_dp) >= 10):
            qual_flag = 'HIGH_SNV'

        elif (p_dn > 0.5) and (child_AD > 0.3):
            qual_flag = 'MEDIUM_SNV'
        elif (p_dn > 0.5) and (child_AD > 0.2) and (variant_AC == '1'):
            qual_flag = 'MEDIUM_SNV'
        elif (p_dn > 0.05) and (child_AD > 0.2):
            qual_flag = 'LOW_SNV'

    return qual_flag


def get_variant_freq(esp_chr_counts, chr_pos_change, variant_annotation):
    "Determine the frequency of alternative alleles at the site"
    esp_freq = 0.0
    vcf_freq = 0.0

    # Determine allele frequency if found in ESP
    if chr_pos_change in esp_chr_counts.keys():
        # sys.stderr.write('Found in ESP: {0}\n'.format(chr_pos_change))
        esp_freq = esp_chr_counts[chr_pos_change]

    # Determine VCF allele frequency (divide AC-1 by AN)
    found_counter = 0
    all_annotation = variant_annotation.split(';')
    for entry in all_annotation:
        entry = entry.split('=')
        if entry[0] == 'AC':
            allele_count = float(entry[1])
            found_counter += 1
        elif entry[0] == 'AN':
            allele_num = float(entry[1])
            found_counter += 1

        if found_counter >= 2:
            break

    try:
        vcf_freq = (allele_count - 1) / allele_num
    except UnboundLocalError:
        sys.exit('What is wrong: {0}\n'.format(variant_annotation))

    # If both esp_freq and vcf_freq are 0, f = 100/30Mbp
    if (esp_freq == 0.0) and (vcf_freq == 0.0):
        return (100.0 / 30000000)

    # Return the greater of the two allele frequencies
    if esp_freq < vcf_freq:
        return vcf_freq
    else:
        return esp_freq


def transform_PL_to_prob(PLs):
    "Take the list of PLs and transform into probability of observing variant"
    # PLs are weird and have to be adjusted like so:
    # P_ref = 10^(-PLref/10)/(10^(-PLref/10) + 10^(-PLhet/10) + 10^(-PLalt/10))

    adj_PL_ref = 10 ** (-float(PLs[0]) / 10)
    adj_PL_het = 10 ** (-float(PLs[1]) / 10)
    adj_PL_alt = 10 ** (-float(PLs[2]) / 10)

    sum_adj_PLs = adj_PL_ref + adj_PL_het + adj_PL_alt

    P_ref = adj_PL_ref / sum_adj_PLs
    P_het = adj_PL_het / sum_adj_PLs
    P_alt = adj_PL_alt / sum_adj_PLs

    return (P_ref, P_het, P_alt)


def get_prob_true_dn(child_PL, dad_PL, mom_PL, variant_pop_freq):
    '''Determine the relative probabilities of a true de novo vs missed het in
    parents'''
    # metric = p(de novo|data)/(p(de novo|data) + p(missed het in parent|data))
    # p(de novo|data) = P_dadref*P_momref*P_kidhet*(1/30Mb)
    # p(mhip|data) = ((P_dadhet*Pmomref + P_momhet*Pdadref)*P_kidhet)
    #                   * (1-(1-f)^4)

    # Fist step: transform all the PLs back into probabilities
    (child_P_ref, child_P_het, child_P_alt) = transform_PL_to_prob(child_PL)
    (dad_P_ref, dad_P_het, dad_P_alt) = transform_PL_to_prob(dad_PL)
    (mom_P_ref, mom_P_het, mom_P_alt) = transform_PL_to_prob(mom_PL)

    # Determine p(de novo | data) -- 1 in 30Mbp is what we expect for dn rate
    p_dn_data = dad_P_ref * mom_P_ref * child_P_het * (1.0 / 30000000)

    # Determine p(missed het in parent | data) -- split for clarity
    p_data_onehet = (dad_P_het * mom_P_ref + dad_P_ref * mom_P_het) * child_P_het
    p_oneparent_het = 1 - ((1 - variant_pop_freq) ** 4)
    p_mhip_data = p_data_onehet * p_oneparent_het

    # Determine the new metric
    metric = p_dn_data / (p_dn_data + p_mhip_data)

    return metric


def process_autosome_variant(line, Fam, PL_pos, AD_pos, DP_pos, args,
                             esp_chr_counts, labels, chrom_under_study,
                             am_kid, who_dad, who_mom, vep_field_names):
    "Go through the VCF line by line to find de novo variants"
    for column, entry in enumerate(line):
        if not entry.startswith(('0/1', '1/0')):
            continue

        # If a het site has been found, check if the het is a child
        if am_kid[column - 9] == 'N':
            continue
        else:
            dad_pos = who_dad[column - 9]
            mom_pos = who_mom[column - 9]

        # Make sure the het variant passes the quality filters
        child_data = child_cuts(entry.split(':'), PL_pos, AD_pos, args)
        if child_data is None:
            continue
        else:
            (child_PL, child_AD_ratio) = child_data

        # Check that the parents are both homozygous reference
        if not line[dad_pos].startswith('0/0'):
            continue
        if not line[mom_pos].startswith('0/0'):
            continue

        dad_record = line[dad_pos].split(':')
        mom_record = line[mom_pos].split(':')

        dad_PL = dad_record[PL_pos].split(',')
        mom_PL = mom_record[PL_pos].split(',')
        dad_AD_info = dad_record[AD_pos]
        mom_AD_info = mom_record[AD_pos]

        # Make sure that both parent genotypes pass AD and DP filters
        parent_AD_ratios = parent_AD_cuts(dad_AD_info, mom_AD_info, args)
        if parent_AD_ratios is None:
            continue
        else:
            (dad_AD_ratio, mom_AD_ratio) = parent_AD_ratios

        child_dp = entry.split(':')[DP_pos]

        DP = [child_dp,
              dad_record[DP_pos],
              mom_record[DP_pos]]

        dp_ratio = DPcheck(DP, args)

        if dp_ratio is None:
            continue

        # Start of the new block with allele frequencies
        variant_position = line[1]
        ref_allele = line[3]
        alt_allele = line[4]
        variant_annotation = line[7]
        chr_pos_change = '{0}:{1}:{2}:{3}'.format(chrom_under_study,
                                                  variant_position,
                                                  ref_allele, alt_allele)

        # Find the population frequency of the variant
        # Max of ESP frequency and frequency in the VCF
        variant_pop_freq = get_variant_freq(esp_chr_counts,
                                            chr_pos_change,
                                            variant_annotation)
        if variant_pop_freq == 0:
            variant_pop_freq = (100.0 / 30000000)
            # Rough expected number of het sites not in ESP

        # Establish the chance that this is a true de novo event
        # using PL scores and variant population frequency
        prob_true_dn = get_prob_true_dn(child_PL, dad_PL, mom_PL,
                                        variant_pop_freq)

        if prob_true_dn < args.pdnmetric:
            continue

        # Extract child sex and affected status
        (dad_col, mom_col, child_sex, child_aff_status) = Fam[column]

        qual_flag = determine_validation_likelihood(ref_allele, alt_allele,
                                                    child_AD_ratio,
                                                    variant_annotation,
                                                    prob_true_dn, dp_ratio,
                                                    child_dp)

        if args.annotatevar:
            (var_gene, var_category) = snpeff_annotate(variant_annotation)
            res_indiv = [
                line[0], line[1], line[2], line[3], line[4],
                labels[column], labels[dad_pos], labels[mom_pos],
                child_sex, child_aff_status, child_PL[0], dad_PL[1],
                mom_PL[1], child_AD_ratio, dad_AD_ratio, mom_AD_ratio,
                DP[0], DP[1], DP[2], dp_ratio, prob_true_dn, var_gene,
                var_category, qual_flag, variant_annotation
            ]
        elif args.annotatevar_VEP:
            (var_gene, var_category) = VEP_annotate(variant_annotation, vep_field_names,
                                                    line[4])  # alt allele also provided
            res_indiv = [
                line[0], line[1], line[2], line[3], line[4],
                labels[column], labels[dad_pos], labels[mom_pos],
                child_sex, child_aff_status, child_PL[0], dad_PL[1],
                mom_PL[1], child_AD_ratio, dad_AD_ratio, mom_AD_ratio,
                DP[0], DP[1], DP[2], dp_ratio, prob_true_dn, var_gene,
                var_category, qual_flag, variant_annotation
            ]
        else:
            res_indiv = [
                line[0], line[1], line[2], line[3], line[4],
                labels[column], labels[dad_pos], labels[mom_pos],
                child_sex, child_aff_status, child_PL[0], dad_PL[1],
                mom_PL[1], child_AD_ratio, dad_AD_ratio, mom_AD_ratio,
                DP[0], DP[1], DP[2], dp_ratio, prob_true_dn, qual_flag,
                line[7]
            ]

        print('\t'.join(map(str, res_indiv)))


def process_multi_variant(line, Fam, PL_pos, AD_pos, DP_pos, args,
                          esp_chr_counts, labels, chrom_under_study,
                          am_kid, who_dad, who_mom, vep_field_names):
    '''Go through the VCF line by line to find de novo variants on lines with
    multiple alt alleles'''
    for column, entry in enumerate(line):
        if not entry.startswith(('0/1', '0/2')):
            continue

        # If a het site has been found, check if the het is a child
        if am_kid[column - 9] == 'N':
            continue
        else:
            dad_pos = who_dad[column - 9]
            mom_pos = who_mom[column - 9]

        # Check that the parents are both homozygous reference -- logic moved
        if not line[dad_pos].startswith('0/0'):
            continue
        if not line[mom_pos].startswith('0/0'):
            continue

        dad_record = line[dad_pos].split(':')
        mom_record = line[mom_pos].split(':')

        # Test if the PL is there
        try:
            d_PL = dad_record[PL_pos].split(',')
        except IndexError:
            continue

        # Replacing the PL and AD information for 0/2
        # Before 0/2 : R,A1,A2 : DP : GQ : RR,RA1,A1A1,RA2,A1A2,A2A2
        # After 0/2 : R,A2 : DP : GQ : RR,RA2,A2A2
        if entry.startswith('0/2'):
            # Fix child
            new_entry = entry.split(':')
            k_AD = new_entry[AD_pos].split(',')
            new_entry[AD_pos] = ','.join([k_AD[0], k_AD[2]])
            k_PL = new_entry[PL_pos].split(',')
            new_entry[PL_pos] = ','.join([k_PL[0], k_PL[3], k_PL[5]])
            entry = ':'.join(new_entry)

            # Fix dad
            if dad_record[AD_pos] == '.':
                dad_record[AD_pos] = '{0},0'.format(dad_record[DP_pos])
            else:
                d_AD = dad_record[AD_pos].split(',')
                dad_record[AD_pos] = ','.join([d_AD[0], d_AD[2]])
            d_PL = dad_record[PL_pos].split(',')
            dad_record[PL_pos] = ','.join([d_PL[0], d_PL[3], d_PL[5]])

            # Fix mom
            if mom_record[AD_pos] == '.':
                mom_record[AD_pos] = '{0},0'.format(mom_record[DP_pos])
            else:
                m_AD = mom_record[AD_pos].split(',')
                mom_record[AD_pos] = ','.join([m_AD[0], m_AD[2]])
            m_PL = mom_record[PL_pos].split(',')
            mom_record[PL_pos] = ','.join([m_PL[0], m_PL[3], m_PL[5]])

        # Make sure the het variant passes the quality filters
        child_data = child_cuts(entry.split(':'), PL_pos, AD_pos, args)
        if child_data is None:
            continue
        else:
            (child_PL, child_AD_ratio) = child_data

        dad_PL = dad_record[PL_pos].split(',')
        mom_PL = mom_record[PL_pos].split(',')
        dad_AD_info = dad_record[AD_pos]
        mom_AD_info = mom_record[AD_pos]

        # Make sure that both parent genotypes pass AD and DP filters
        parent_AD_ratios = parent_AD_cuts(dad_AD_info, mom_AD_info, args)
        if parent_AD_ratios is None:
            continue
        else:
            (dad_AD_ratio, mom_AD_ratio) = parent_AD_ratios

        child_dp = entry.split(':')[DP_pos]

        DP = [child_dp,
              dad_record[DP_pos],
              mom_record[DP_pos]]

        dp_ratio = DPcheck(DP, args)

        if dp_ratio is None:
            continue

        # Start of the new block with allele frequencies
        variant_position = line[1]
        ref_allele = line[3]
        alt_alleles = line[4].split(',')
        variant_annotation_s = line[7].split(';')
        v_AC = variant_annotation_s[0].split(',')
        # AC should be the 1st position of annotation, "AC=1,2" -> "AC=1" and "2"

        if entry.startswith('0/2'):
            alt_allele = alt_alleles[1]
            variant_annotation_s[0] = 'AC=' + v_AC[1]  # to make "AC=2"
        else:
            alt_allele = alt_alleles[0]
            variant_annotation_s[0] = v_AC[0]  # should be "AC=1"

        variant_annotation = ';'.join(variant_annotation_s)
        chr_pos_change = '{0}:{1}:{2}:{3}'.format(chrom_under_study,
                                                  variant_position,
                                                  ref_allele, alt_allele)

        # Find the population frequency of the variant
        # Max of ESP frequency and frequency in the VCF
        variant_pop_freq = get_variant_freq(esp_chr_counts,
                                            chr_pos_change,
                                            variant_annotation)
        if variant_pop_freq == 0:
            variant_pop_freq = (100.0 / 30000000)
            # Rough expected number of het sites not in ESP

        # Establish the chance that this is a true de novo event
        # using PL scores and variant population frequency
        prob_true_dn = get_prob_true_dn(child_PL, dad_PL, mom_PL,
                                        variant_pop_freq)

        if prob_true_dn < args.pdnmetric:
            continue

        # Extract child sex and affected status
        (dad_col, mom_col, child_sex, child_aff_status) = Fam[column]

        qual_flag = determine_validation_likelihood(ref_allele, alt_allele,
                                                    child_AD_ratio,
                                                    variant_annotation, prob_true_dn,
                                                    dp_ratio, child_dp)

        if args.annotatevar:
            (var_gene, var_category) = snpeff_annotate(variant_annotation)
            if ',' in var_gene:
                var_genes = var_gene.split(',')
                if entry.startswith('0/2'):
                    var_gene = var_genes[1]
                else:
                    var_gene = var_genes[0]
            elif ',' in var_category:
                var_cats = var_category.split(',')
                if entry.startswith('0/2'):
                    var_category = var_cats[1]
                else:
                    var_category = var_cats[0]

            res_indiv = [
                line[0], line[1], line[2], line[3], alt_allele,
                labels[column], labels[dad_pos], labels[mom_pos],
                child_sex, child_aff_status, child_PL[0], dad_PL[1],
                mom_PL[1], child_AD_ratio, dad_AD_ratio, mom_AD_ratio,
                DP[0], DP[1], DP[2], dp_ratio, prob_true_dn, var_gene,
                var_category, qual_flag, line[7]
            ]
        elif args.annotatevar_VEP:
            (var_gene, var_category) = VEP_annotate(variant_annotation, vep_field_names,
                                                    alt_allele)  # alt allele also provided
            if ',' in var_gene:
                var_genes = var_gene.split(',')
                if entry.startswith('0/2'):
                    var_gene = var_genes[1]
                else:
                    var_gene = var_genes[0]
            elif ',' in var_category:
                var_cats = var_category.split(',')
                if entry.startswith('0/2'):
                    var_category = var_cats[1]
                else:
                    var_category = var_cats[0]

            res_indiv = [
                line[0], line[1], line[2], line[3], alt_allele,
                labels[column], labels[dad_pos], labels[mom_pos],
                child_sex, child_aff_status, child_PL[0], dad_PL[1],
                mom_PL[1], child_AD_ratio, dad_AD_ratio, mom_AD_ratio,
                DP[0], DP[1], DP[2], dp_ratio, prob_true_dn, var_gene,
                var_category, qual_flag, line[7]
            ]
        else:
            res_indiv = [
                line[0], line[1], line[2], line[3], alt_allele,
                labels[column], labels[dad_pos], labels[mom_pos],
                child_sex, child_aff_status, child_PL[0], dad_PL[1],
                mom_PL[1], child_AD_ratio, dad_AD_ratio, mom_AD_ratio,
                DP[0], DP[1], DP[2], dp_ratio, prob_true_dn, qual_flag,
                line[7]
            ]

        print('\t'.join(map(str, res_indiv)))


def process_hemizygous_variants(line, Fam, PL_pos, AD_pos, DP_pos, args,
                                esp_chr_counts, labels, chrom_under_study,
                                parent, gender_kid, who_parent, vep_field_names):
    "Look for de novo variants when the chromosome is hemizygous"
    for column, entry in enumerate(line):
        if not entry.startswith('1/1'):
            continue

        # Check if the column is a kid's
        if gender_kid[column - 9] == 'N':
            continue
        else:
            par_pos = who_parent[column - 9]

        # Only keep lines where and parents are ref
        if not line[par_pos].startswith('0/0'):
            continue

        child_record = entry.split(':')
        par_record = line[par_pos].split(':')

        # Expect child to be homozygous alternative
        child_PL = child_record[PL_pos].split(',')
        if child_PL[2] != '0':
            continue
        if int(child_PL[1]) <= args.thresh:
            continue

        child_AD = child_record[AD_pos].split(',')
        if ((child_AD[0] == '0') and (child_AD[1] == '0')):
            continue

        child_AD_ratio = float(child_AD[1]) / (float(child_AD[0]) +
                                               float(child_AD[1]))
        if child_AD_ratio <= 0.95:
            continue

        # Check that parent's reads match homozygous reference
        if par_record[AD_pos] == '.':
            par_AD_ratio = 0.0
        else:
            par_AD = par_record[AD_pos].split(',')
            if ((par_AD[0] == '0') and (par_AD[1] == '0')):
                continue

            par_AD_ratio = float(par_AD[1]) / (float(par_AD[0]) +
                                               float(par_AD[1]))
            if par_AD_ratio >= 0.05:
                continue

        par_PL = par_record[PL_pos].split(',')

        # Depth filter
        child_dp = child_record[DP_pos]

        dp_ratio = float(child_dp) / float(par_record[DP_pos])
        percent_depthratio = args.depthratio / 100.0
        if dp_ratio <= percent_depthratio:
            continue

        # Find variant information
        variant_position = line[1]
        ref_allele = line[3]
        alt_allele = line[4]
        variant_annotation = line[7]
        chr_pos_change = '{0}:{1}:{2}:{3}'.format(chrom_under_study,
                                                  variant_position,
                                                  ref_allele, alt_allele)

        # Establish the chance that this is a true de novo event
        # using PL scores and variant population frequency
        # Fist step: transform all the PLs back into probabilities
        (child_P_ref, child_P_het, child_P_alt) = transform_PL_to_prob(child_PL)
        (par_P_ref, par_P_het, par_P_alt) = transform_PL_to_prob(par_PL)

        # Determine p(de novo | data) and part of p(missed alt | data)
        p_dn_data = par_P_ref * child_P_alt * (1.0 / 30000000)
        p_data_missedcall = (par_P_het + par_P_alt) * child_P_alt

        # Find the population frequency of the variant
        # Max of ESP frequency and frequency in the VCF
        variant_pop_freq = get_variant_freq(esp_chr_counts,
                                            chr_pos_change,
                                            variant_annotation)
        if variant_pop_freq == 0:
            variant_pop_freq = (100.0 / 30000000)
            # Rough expected number of het sites not in ESP

        # Determine the new metric and remove unlikely de novo variants
        p_oneparent_het = 1 - ((1 - variant_pop_freq) ** 4)
        p_mhip_data = p_data_missedcall * p_oneparent_het
        prob_true_dn = p_dn_data / (p_dn_data + p_mhip_data)
        if prob_true_dn < args.pdnmetric:
            continue

        # Getting other parent information
        # need dad_PL, mom_PL, dad_AD_ratio, mom_AD_ratio, depths
        if parent == 'mom':
            dad_PL = ['.', '.', '.']
            dad_AD_ratio = '.'
            dad_DP = '.'
            mom_PL = par_PL
            mom_AD_ratio = par_AD_ratio
            mom_DP = par_record[DP_pos]
        elif parent == 'dad':
            mom_PL = ['.', '.', '.']
            mom_AD_ratio = '.'
            mom_DP = '.'
            dad_PL = par_PL
            dad_AD_ratio = par_AD_ratio
            dad_DP = par_record[DP_pos]
        else:
            continue

        qual_flag = determine_validation_likelihood(ref_allele, alt_allele,
                                                    child_AD_ratio,
                                                    variant_annotation, prob_true_dn,
                                                    dp_ratio, child_dp)

        (dad_pos, mom_pos, gender, aff_status) = Fam[column]

        # Annotate and print
        if args.annotatevar:
            (var_gene, var_category) = snpeff_annotate(variant_annotation)
            res_indiv = [
                line[0], line[1], line[2], line[3], line[4],
                labels[column], labels[dad_pos], labels[mom_pos],
                gender, aff_status, child_PL[0], dad_PL[1], mom_PL[1],
                child_AD_ratio, dad_AD_ratio, mom_AD_ratio, child_record[DP_pos],
                dad_DP, mom_DP, dp_ratio, prob_true_dn, var_gene, var_category,
                qual_flag, variant_annotation
            ]
        elif args.annotatevar_VEP:
            (var_gene, var_category) = VEP_annotate(variant_annotation, vep_field_names,
                                                    line[4])  # alt allele also provided
            res_indiv = [
                line[0], line[1], line[2], line[3], line[4],
                labels[column], labels[dad_pos], labels[mom_pos],
                gender, aff_status, child_PL[0], dad_PL[1], mom_PL[1],
                child_AD_ratio, dad_AD_ratio, mom_AD_ratio, child_record[DP_pos],
                dad_DP, mom_DP, dp_ratio, prob_true_dn, var_gene, var_category,
                qual_flag, variant_annotation
            ]
        else:
            res_indiv = [
                line[0], line[1], line[2], line[3], line[4],
                labels[column], labels[dad_pos], labels[mom_pos],
                gender, aff_status, child_PL[0], dad_PL[1], mom_PL[1],
                child_AD_ratio, dad_AD_ratio, mom_AD_ratio, child_record[DP_pos],
                dad_DP, mom_DP, dp_ratio, prob_true_dn, qual_flag, line[7]
            ]

        print('\t'.join(map(str, res_indiv)))


def process_multi_hemi_variant(line, Fam, PL_pos, AD_pos, DP_pos, args,
                               esp_chr_counts, labels, chrom_under_study,
                               parent, gender_kid, who_parent, vep_field_names):
    '''Go through the VCF line by line to find de novo variants on lines with
    multiple alt alleles for hemizygous chromosomes'''
    for column, entry in enumerate(line):
        if not entry.startswith(('1/1', '2/2')):
            continue

        # Check if the column is a kid's
        if gender_kid[column - 9] == 'N':
            continue
        else:
            par_pos = who_parent[column - 9]

        # Check that the parent is homozygous reference
        if not line[par_pos].startswith('0/0'):
            continue

        child_record = entry.split(':')
        par_record = line[par_pos].split(':')

        # Replacing the PL and AD information for 2/2
        # Before 2/2 : R,A1,A2 : DP : GQ : RR,RA1,A1A1,RA2,A1A2,A2A2
        # After 2/2 : R,A2 : DP : GQ : RR,RA2,A2A2
        if entry.startswith('2/2'):
            # Fix child
            k_AD = child_record[AD_pos].split(',')
            child_record[AD_pos] = ','.join([k_AD[0], k_AD[2]])
            k_PL = child_record[PL_pos].split(',')
            child_record[PL_pos] = ','.join([k_PL[0], k_PL[3], k_PL[5]])

            # Fix parent
            if par_record[AD_pos] == '.':
                par_record[AD_pos] = '{0},0'.format(par_record[DP_pos])
            else:
                p_AD = par_record[AD_pos].split(',')
                par_record[AD_pos] = ','.join([p_AD[0], p_AD[2]])
            p_PL = par_record[PL_pos].split(',')
            par_record[PL_pos] = ','.join([p_PL[0], p_PL[3], p_PL[5]])

        # Expect child to be homozygous alternative
        child_PL = child_record[PL_pos].split(',')
        if child_PL[2] != '0':
            continue
        if int(child_PL[1]) <= args.thresh:
            continue

        child_AD = child_record[AD_pos].split(',')
        if ((child_AD[0] == '0') and (child_AD[1] == '0')):
            continue

        child_AD_ratio = float(child_AD[1]) / (float(child_AD[0]) +
                                               float(child_AD[1]))
        if child_AD_ratio <= 0.95:
            continue

        # Check that parent's reads match homozygous reference
        if par_record[AD_pos] == '.':
            par_AD_ratio = 0.0
        else:
            par_AD = par_record[AD_pos].split(',')
            if ((par_AD[0] == '0') and (par_AD[1] == '0')):
                continue

            par_AD_ratio = float(par_AD[1]) / (float(par_AD[0]) +
                                               float(par_AD[1]))
            if par_AD_ratio >= 0.05:
                continue

        par_PL = par_record[PL_pos].split(',')

        # Depth filter
        child_dp = child_record[DP_pos]

        dp_ratio = float(child_dp) / float(par_record[DP_pos])
        percent_depthratio = args.depthratio / 100.0
        if dp_ratio <= percent_depthratio:
            continue

        # Start of the new block with allele frequencies
        variant_position = line[1]
        ref_allele = line[3]
        alt_alleles = line[4].split(',')
        variant_annotation_s = line[7].split(';')
        v_AC = variant_annotation_s[0].split(',')
        # AC should be the 1st position of annotation, "AC=1,2" -> "AC=1" and "2"

        if entry.startswith('2/2'):
            alt_allele = alt_alleles[1]
            variant_annotation_s[0] = 'AC=' + v_AC[1]  # to make "AC=2"
        else:
            alt_allele = alt_alleles[0]
            variant_annotation_s[0] = v_AC[0]  # should be "AC=1"

        variant_annotation = ';'.join(variant_annotation_s)
        chr_pos_change = '{0}:{1}:{2}:{3}'.format(chrom_under_study,
                                                  variant_position,
                                                  ref_allele, alt_allele)

        # Establish the chance that this is a true de novo event
        # using PL scores and variant population frequency
        # Fist step: transform all the PLs back into probabilities
        (child_P_ref, child_P_het, child_P_alt) = transform_PL_to_prob(child_PL)
        (par_P_ref, par_P_het, par_P_alt) = transform_PL_to_prob(par_PL)

        # Determine p(de novo | data) and part of p(missed alt | data)
        p_dn_data = par_P_ref * child_P_alt * (1.0 / 30000000)
        p_data_missedcall = (par_P_het + par_P_alt) * child_P_alt

        # Find the population frequency of the variant
        # Max of ESP frequency and frequency in the VCF
        variant_pop_freq = get_variant_freq(esp_chr_counts,
                                            chr_pos_change,
                                            variant_annotation)
        if variant_pop_freq == 0:
            variant_pop_freq = (100.0 / 30000000)
            # Rough expected number of het sites not in ESP

        # Determine the new metric and remove unlikely de novo variants
        p_oneparent_het = 1 - ((1 - variant_pop_freq) ** 4)
        p_mhip_data = p_data_missedcall * p_oneparent_het
        prob_true_dn = p_dn_data / (p_dn_data + p_mhip_data)
        if prob_true_dn < args.pdnmetric:
            continue

        # Getting other parent information
        # need dad_PL, mom_PL, dad_AD_ratio, mom_AD_ratio, depths
        if parent == 'mom':
            dad_PL = ['.', '.', '.']
            dad_AD_ratio = '.'
            dad_DP = '.'
            mom_PL = par_PL
            mom_AD_ratio = par_AD_ratio
            mom_DP = par_record[DP_pos]
        elif parent == 'dad':
            mom_PL = ['.', '.', '.']
            mom_AD_ratio = '.'
            mom_DP = '.'
            dad_PL = par_PL
            dad_AD_ratio = par_AD_ratio
            dad_DP = par_record[DP_pos]
        else:
            continue

        qual_flag = determine_validation_likelihood(ref_allele, alt_allele,
                                                    child_AD_ratio,
                                                    variant_annotation, prob_true_dn,
                                                    dp_ratio, child_dp)

        (dad_pos, mom_pos, gender, aff_status) = Fam[column]

        # Annotate and print
        if args.annotatevar:
            (var_gene, var_category) = snpeff_annotate(variant_annotation)
            if ',' in var_gene:
                var_genes = var_gene.split(',')
                if entry.startswith('2/2'):
                    var_gene = var_genes[1]
                else:
                    var_gene = var_genes[0]
            elif ',' in var_category:
                var_cats = var_category.split(',')
                if entry.startswith('2/2'):
                    var_category = var_cats[1]
                else:
                    var_category = var_cats[0]

            res_indiv = [
                line[0], line[1], line[2], line[3], alt_allele,
                labels[column], labels[dad_pos], labels[mom_pos],
                gender, aff_status, child_PL[0], dad_PL[1], mom_PL[1],
                child_AD_ratio, dad_AD_ratio, mom_AD_ratio, child_record[DP_pos],
                dad_DP, mom_DP, dp_ratio, prob_true_dn, var_gene, var_category,
                qual_flag, line[7]
            ]
        elif args.annotatevar_VEP:
            (var_gene, var_category) = VEP_annotate(variant_annotation, vep_field_names,
                                                    alt_allele)  # alt allele also provided
            if ',' in var_gene:
                var_genes = var_gene.split(',')
                if entry.startswith('2/2'):
                    var_gene = var_genes[1]
                else:
                    var_gene = var_genes[0]
            elif ',' in var_category:
                var_cats = var_category.split(',')
                if entry.startswith('2/2'):
                    var_category = var_cats[1]
                else:
                    var_category = var_cats[0]

            res_indiv = [
                line[0], line[1], line[2], line[3], alt_allele,
                labels[column], labels[dad_pos], labels[mom_pos],
                gender, aff_status, child_PL[0], dad_PL[1], mom_PL[1],
                child_AD_ratio, dad_AD_ratio, mom_AD_ratio, child_record[DP_pos],
                dad_DP, mom_DP, dp_ratio, prob_true_dn, var_gene, var_category,
                qual_flag, line[7]
            ]
        else:
            res_indiv = [
                line[0], line[1], line[2], line[3], alt_allele,
                labels[column], labels[dad_pos], labels[mom_pos],
                gender, aff_status, child_PL[0], dad_PL[1], mom_PL[1],
                child_AD_ratio, dad_AD_ratio, mom_AD_ratio, child_record[DP_pos],
                dad_DP, mom_DP, dp_ratio, prob_true_dn, qual_flag, line[7]
            ]

        print('\t'.join(map(str, res_indiv)))


def main(vcffile, Fam, args):
    "Run the checks on each line in the VCF"

    # Printing a warning about tri-allelic sites
    sys.stderr.write('*** WARNGING: Only processing sites up to triallelic (one ref, two alt) ***\n')

    # Printing out script and run information
    print('## Run start date and time: {0}'.format(time.strftime("%c")))  # date and time
    print('## Script version: {0}'.format(__version__))  # script version used
    print('## Command given: python {0}'.format(' '.join(sys.argv)))  # full command

    # Printing labels line
    if args.annotatevar | args.annotatevar_VEP:
        print('\t'.join([
            "Chr", "Pos", "rsID", "Ref", "Alt", "Child_ID", "Dad_ID",
            "Mom_ID", "Child_Sex", "Child_AffectedStatus", "Child_PL_AA",
            "Dad_PL_AB", "Mom_PL_AB", "Child_AD_Ratio", "Dad_AD_Ratio",
            "Mom_AD_Ratio", "DP_Child", "DP_Dad", "DP_Mom", "DP_Ratio",
            "Prob_dn", "Gene_name", "Category", "Validation_Likelihood",
            "Annotation"
        ]))
    else:
        print('\t'.join([
            "Chr", "Pos", "rsID", "Ref", "Alt", "Child_ID", "Dad_ID",
            "Mom_ID", "Child_Sex", "Child_AffectedStatus", "Child_PL_AA",
            "Dad_PL_AB", "Mom_PL_AB", "Child_AD_Ratio", "Dad_AD_Ratio",
            "Mom_AD_Ratio", "DP_Child", "DP_Dad", "DP_Mom", "DP_Ratio",
            "Prob_dn", "Validation_Likelihood", "Annotation"
        ]))

    # Reading header lines to get VEP and individual arrays
    vep_field_names = '.'  # holder, should be replaced if VEP annotation is present

    header_line = vcffile.next()
    while header_line.startswith('##'):
        if header_line.find('ID=CSQ') > -1:
            vep_field_names = header_line.split('Format: ')[-1].strip('">').split('|')
        header_line = vcffile.next()

    if not header_line.startswith('#CHROM'):  # should be the #CHROM line
        sys.stderr.write('ERROR: Unexpected header line: Expected line starting with "#CHROM"'
                         '\n  Found: {0}...'.format(header_line[:30]))
        sys.exit(1)

    labels = header_line.strip().split('\t')

    (Fam, am_kid, who_dad, who_mom) = trimfamily(Fam, labels)
    (female_Fam, female_kid, male_Fam, male_kid) = split_Fam(Fam, labels)

    no_PL_lines = 0
    no_AD_lines = 0
    current_chrom = '1'

    # Only loading the ESP counts for the current chromosome
    sys.stderr.write('Loading chr{0} counts...\n'.format(current_chrom))
    esp_chr_counts = load_esp_counts(args.esp, current_chrom)
    sys.stderr.write('\tFinished loading chr{0} counts\n'.format(
            current_chrom))

    # Move line by line through the VCF
    for line in vcffile:
        line = line.strip('\n').split('\t')

        chrom_under_study = line[0]
        if chrom_under_study.startswith('chr'):
            chrom_under_study = chrom_under_study[3:]

        # Update the count dictionary if this line has a new chromosome
        if chrom_under_study != current_chrom:
            current_chrom = chrom_under_study
            sys.stderr.write('Loading chr{0} counts...\n'.format(
                    current_chrom))
            esp_chr_counts = load_esp_counts(args.esp, current_chrom)
            sys.stderr.write('\tFinished loading chr{0} counts\n'.format(
                    current_chrom))

        if not process_line(line, args):  # removes non-PASSing lines
            continue

        format_line = line[8].split(':')
        try:
            PL_pos = format_line.index('PL')
        except ValueError:
            no_PL_lines += 1  # Added to count lines missing PL
            continue

        try:
            AD_pos = format_line.index('AD')
        except ValueError:
            no_AD_lines += 1  # Added to count lines missing AD
            continue

        DP_pos = format_line.index('DP')

        # Finding multi-allelic lines that should be processed (no indels)
        multi_flag = False
        ref_allele = line[3]
        if ',' in ref_allele:  # I don't think this should happen
            continue

        alt_allele = line[4]
        if ',' in alt_allele:
            alt_alleles = alt_allele.split(',')
            multi_flag = True

            # Removing sites with more than 2 alternative alleles. Script cannot process them at this time.
            if len(alt_alleles) > 2:
                continue

        # Treating sex chromosomes differently
        if chrom_under_study in ('x', 'X', '23'):
            chrom_under_study = 'X'
            if ((int(line[1]) < 2699520) or
                    (int(line[1]) > 154931044)):  # pseudoautosomal regions
                if multi_flag == True:
                    process_multi_variant(line, Fam, PL_pos, AD_pos, DP_pos, args,
                                          esp_chr_counts, labels,
                                          chrom_under_study, am_kid, who_dad, who_mom, vep_field_names)

                else:
                    process_autosome_variant(line, Fam, PL_pos, AD_pos,
                                             DP_pos, args, esp_chr_counts,
                                             labels, chrom_under_study, am_kid,
                                             who_dad, who_mom, vep_field_names)
            else:
                if len(female_Fam) > 0:
                    if multi_flag == True:
                        process_multi_variant(line, female_Fam, PL_pos, AD_pos, DP_pos,
                                              args, esp_chr_counts, labels,
                                              chrom_under_study, female_kid, who_dad,
                                              who_mom, vep_field_names)
                    else:
                        process_autosome_variant(line, female_Fam, PL_pos, AD_pos,
                                                 DP_pos, args, esp_chr_counts,
                                                 labels, chrom_under_study,
                                                 female_kid, who_dad, who_mom, vep_field_names)

                if len(male_Fam) > 0:
                    if multi_flag == True:
                        process_multi_hemi_variant(line, male_Fam, PL_pos, AD_pos, DP_pos,
                                                   args, esp_chr_counts, labels,
                                                   chrom_under_study, 'mom', male_kid,
                                                   who_mom, vep_field_names)
                    else:
                        process_hemizygous_variants(line, male_Fam, PL_pos, AD_pos,
                                                    DP_pos, args, esp_chr_counts,
                                                    labels, chrom_under_study,
                                                    'mom', male_kid, who_mom, vep_field_names)

        elif chrom_under_study in ('y', 'Y', '24'):
            chrom_under_study = 'Y'
            if multi_flag == True:
                process_multi_hemi_variant(line, male_Fam, PL_pos, AD_pos, DP_pos,
                                           args, esp_chr_counts, labels,
                                           chrom_under_study, 'dad', male_kid,
                                           who_dad, vep_field_names)
            else:
                process_hemizygous_variants(line, male_Fam, PL_pos, AD_pos,
                                            DP_pos, args, esp_chr_counts,
                                            labels, chrom_under_study, 'dad', male_kid,
                                            who_dad, vep_field_names)

        # Autosomes
        else:
            if multi_flag == True:
                process_multi_variant(line, Fam, PL_pos, AD_pos, DP_pos, args,
                                      esp_chr_counts, labels,
                                      chrom_under_study, am_kid, who_dad, who_mom, vep_field_names)
            else:
                process_autosome_variant(line, Fam, PL_pos, AD_pos, DP_pos, args,
                                         esp_chr_counts, labels,
                                         chrom_under_study, am_kid, who_dad, who_mom, vep_field_names)

    sys.stderr.write('Number of lines with no PL information: {0}\n'.format(
            no_PL_lines))
    sys.stderr.write('Number of lines with no AD information: {0}\n'.format(
            no_AD_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Find de novo variants in a given vcf
    ''')
    parser.add_argument('vcf', action='store', type=str, help='VCF file')
    parser.add_argument('fam', action='store', type=str,
                        help='Family relations file')
    parser.add_argument('esp', action='store', type=str,
                        help='ESP counts file')

    parser.add_argument('-t', '--thresh', action='store', type=int,
                        dest='thresh', help='PL threshold (default=20)',
                        default=20)
    parser.add_argument('-m', '--prob(dn)metric', action='store', type=float,
                        dest='pdnmetric',
                        help='''Minimum prob(dn) value that a variant needs to
                        be printed (default=0.05''', default=0.05)
    parser.add_argument('-p', '--maxparentAB', action='store', type=float,
                        dest='maxparentAB',
                        help='''Max parent allele balance (nonref/total calls)
                                is less than specified decimal
                                (default=0.05)''',
                        default=0.05)
    parser.add_argument('-c', '--minchildAB', action='store', type=float,
                        dest='minchildAB',
                        help='''Min child allele balance (nonref/total calls)
                                is greater than specified decimal
                                (default=0.2)''',
                        default=0.2)
    parser.add_argument('-d', '--depthratio', action='store', type=int,
                        dest='depthratio',
                        help='''Child depth is at least 1/(given integer) of
                                the sum of the depth in both parents
                                (default=10)''',
                        default=10)

    # Optional flag to annotate the de novo variants found
    parser.add_argument('-a', '--annotatevariants', action='store_true',
                        dest='annotatevar', help='Annotate de novo variants')
    parser.add_argument('-v', '--annotatevariants_VEP', action='store_true',
                        dest='annotatevar_VEP', help='Annotate de novo variants')
    # Note: VEP annotation code is hacked from Konrad Karczewski

    args = parser.parse_args()

    if not os.path.exists(args.vcf):
        sys.exit("{0}: No such file or directory".format(args.vcf))
    if not os.path.exists(args.fam):
        sys.exit("{0}: No such file or directory".format(args.fam))
    if not os.path.exists(args.esp):
        sys.exit("{0}: No such file or directory".format(args.esp))

    with open(args.fam, 'r') as pedfile:
        Fam = {}
        # Expecting family file to have 6 columns: family_id, proband_id,
        # dad_id, mom_id, proband_gender, proband_affected_status

        for line in pedfile:
            line = line.split()
            if (line[2] == '0') and (line[3] == '0'):
                continue
            Fam[line[1]] = [line[2], line[3], line[4], line[5]]
            # Stores kid: [dad, mom, gender, affected]

    if '.gz' in args.vcf:
        vcffile = gzip.open(args.vcf, 'r')
    else:
        vcffile = open(args.vcf, 'r')

    main(vcffile, Fam, args)
