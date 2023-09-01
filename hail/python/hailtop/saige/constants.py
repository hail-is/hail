from enum import Enum
from typing import Dict


class SaigeInputDataType(Enum):
    VCF = 'vcf'
    BGEN = 'bgen'


class SaigeTestType(Enum):
    QUANTITATIVE = 'quantitative'
    BINARY = 'binary'


class SaigePhenotype(Enum):
    CONTINUOUS = 'continuous'
    BIOMARKERS = 'biomarkers'
    CATEGORICAL = 'categorical'
    ICD = 'icd'
    ICD_FIRST_OCCURRENCE = 'icd_first_occurrence'
    ICD_ALL = 'icd_all'
    PHECODE = 'phecode'
    PRESCRIPTIONS = 'prescriptions'


saige_phenotype_to_test_type: Dict[SaigePhenotype, SaigeTestType] = {
    SaigePhenotype.CONTINUOUS: SaigeTestType.QUANTITATIVE,
    SaigePhenotype.BIOMARKERS: SaigeTestType.QUANTITATIVE,
    SaigePhenotype.CATEGORICAL: SaigeTestType.BINARY,
    SaigePhenotype.ICD: SaigeTestType.BINARY,
    SaigePhenotype.ICD_FIRST_OCCURRENCE: SaigeTestType.BINARY,
    SaigePhenotype.ICD_ALL: SaigeTestType.BINARY,
    SaigePhenotype.PHECODE: SaigeTestType.BINARY,
    SaigePhenotype.PRESCRIPTIONS: SaigeTestType.BINARY,
}


class SaigeAnalysisType(Enum):
    VARIANT = 'variant'
    GENE = 'gene'
