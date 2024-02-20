from enum import Enum


class SaigeInputDataType(Enum):
    VCF = 'vcf'
    BGEN = 'bgen'


class SaigeAnalysisType(Enum):
    VARIANT = 'variant'
    GENE = 'gene'
