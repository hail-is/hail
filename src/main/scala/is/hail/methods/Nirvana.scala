package is.hail.methods

import is.hail.expr.{TArray, TBoolean, TDouble, TInt, TString, TStruct}


class Nirvana {
  val nirvanaSignature = TStruct(
    "header" -> TStruct(
      "annotator" -> TString,
      "creationTime" -> TString,
      "schemaVersion" -> TInt,
      "dataVersion" -> TString,
      "dataSources" -> TArray(TStruct(
        "name" -> TString,
        "version" -> TString,
        "description" -> TString,
        "releaseDate" -> TString
      )),
      "genomeAssembly" -> TString,
      "samples" -> TArray(TString)
    ),
    "positions" -> TArray(TStruct(
      "chromosome" -> TString,
      "refAllele" -> TString,
      "position" -> TInt,
      "altAlleles" -> TArray(TString),
      "cytogeneticBand" -> TString,
      "quality" -> TInt,
      "filters" -> TArray(TString),
      "jointSomaticNormalQuality" -> TInt,
      "copyNumber" -> TInt,
      "strandBias" -> TDouble,
      "recalibratedQual ity" -> TDouble,
      "samples" -> TArray(TStruct(
        "variantFreq" -> TDouble,
        "totalDepth" -> TInt,
        "alleleDepths" -> TArray(TInt),
        "genotype" -> TString,
        "genotypeQuality" -> TInt,
        "failedFilter" -> TBoolean,
        "isEmpty" -> TBoolean,
        "copyNumber" -> TInt,
        "lossOfHeterozygosity" -> TBoolean
      )),
      "variants" -> TArray(TStruct(
        "ancestralAllele" -> TString,
        "altAllele" -> TString,
        "refAllele" -> TString,
        "chromosome" -> TString,
        "begin" -> TInt,
        "end" -> TInt,
        "phylopScore" -> TDouble,
        "dbsnp" -> TArray(TString),
        "globalMinorAllele" -> TString,
        "gmaf" -> TDouble,
        "isReferenceMinorAllele" -> TBoolean,
        "variantType" -> TString,
        "vid" -> TString,
        "oneKgAll" -> TDouble,
        "oneKgAllAc" -> TInt,
        "oneKgAllAn" -> TInt,
        "oneKgAfr" -> TDouble,
        "oneKgAfrAc" -> TInt,
        "oneKgAfrAn" -> TInt,
        "oneKgAmr" -> TDouble,
        "oneKgAmrAc" -> TInt,
        "oneKgAmrAn" -> TInt,
        "oneKgEas" -> TDouble,
        "oneKgEasAc" -> TInt,
        "oneKgEasAn" -> TInt,
        "oneKgEur" -> TDouble,
        "oneKgEurAc" -> TInt,
        "oneKgEurAn" -> TInt,
        "oneKgSas" -> TDouble,
        "oneKgSasAc" -> TInt,
        "oneKgSasAn" -> TInt,
        "evsCoverage" -> TInt,
        "evsSamples" -> TInt,
        "evsAll" -> TDouble,
        "evsAfr" -> TDouble,
        "evsEur" -> TDouble,
        "exacCoverage" -> TInt,
        "exacAll" -> TDouble,
        "exacAllAc" -> TInt,
        "exacAllAn" -> TInt,
        "exacAfr" -> TDouble,
        "exacAfrAc" -> TInt,
        "exacAfrAn" -> TInt,
        "exacAmr" -> TDouble,
        "exacAmrAc" -> TInt,
        "exacAmrAn" -> TInt,
        "exacEas" -> TDouble,
        "exacEasAc" -> TInt,
        "exacEasAn" -> TInt,
        "exacFin" -> TDouble,
        "exacFinAc" -> TInt,
        "exacFinAn" -> TInt,
        "exacNfe" -> TDouble,
        "exacNfeAc" -> TInt,
        "exacNfeAn" -> TInt,
        "exacOth" -> TDouble,
        "exacOthAc" -> TInt,
        "exacOthAn" -> TInt,
        "exacSas" -> TDouble,
        "exacSasAc" -> TInt,
        "exacSasAn" -> TInt
      )),
      "regulatoryRegions" -> TArray(TStruct(
        "id" -> TString,
        "consequence" -> TArray(TString)
      )),
      "clinVar" -> TArray(TStruct(
        "id" -> TString,
        "reviewStatus" -> TString,
        "isAlleleSpecific" -> TBoolean,
        "alleleOrigin" -> TString,
        "refAllele" -> TString,
        "altAllele" -> TString,
        "phenotype" -> TString,
        "geneReviewsId" -> TString,
        "medGenId" -> TString,
        "omimId" -> TString,
        "orphanetId" -> TString,
        "significance" -> TString,
        "snoMetCtId" -> TString,
        "lastEvaluatedDate" -> TString,
        "pubMedIds" -> TArray(TString)
      )),
      "cosmic" -> TArray(TStruct(
        "id" -> TString,
        "isAlleleSpecific" -> TBoolean,
        "refAllele" -> TString,
        "altAllele" -> TString,
        "gene" -> TString,
        "studies" -> TArray(TStruct(
          "id" -> TInt,
          "histology" -> TString,
          "primarySite" -> TString
        ))
      )),
      "transcripts" -> TStruct(
        "ensembl" -> TArray(TStruct(
          "transcript" -> TString,
          "aminoAcids" -> TString,
          "bioType" -> TString,
          "cDnaPos" -> TString,
          "codons" -> TString,
          "cdsPos" -> TString,
          "exons" -> TString,
          "introns" -> TString,
          "geneId" -> TString,
          "hgnc" -> TString,
          "consequence" -> TArray(TString),
          "hgvsc" -> TString,
          "hgvsp" -> TString,
          "isCanonical" -> TBoolean,
          "polyPhenScore" -> TDouble,
          "polyPhenPrediction" -> TString,
          "proteinId" -> TString,
          "proteinPos" -> TString,
          "siftScore" -> TDouble,
          "siftPrediction" -> TString
        ))
      )
    ))
  )


}
