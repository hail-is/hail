package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.utils._

object VSMLocalValue {
  def apply(sampleIds: IndexedSeq[String]): VSMLocalValue =
    VSMLocalValue(Annotation.empty,
      sampleIds,
      Annotation.emptyIndexedSeq(sampleIds.length))
}

case class VSMLocalValue(
  globalAnnotation: Annotation,
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotation]) {
  assert(sampleIds.areDistinct(), s"Sample ID names are not distinct: ${ sampleIds.duplicates().mkString(", ") }")
  assert(sampleIds.length == sampleAnnotations.length)

  def nSamples: Int = sampleIds.length

  def dropSamples(): VSMLocalValue = VSMLocalValue(globalAnnotation,
    IndexedSeq.empty[String],
    IndexedSeq.empty[Annotation])
}

object VSMFileMetadata {
  def apply(sampleIds: IndexedSeq[String],
    sampleAnnotations: IndexedSeq[Annotation] = null,
    globalAnnotation: Annotation = Annotation.empty,
    saSignature: Type = TStruct.empty,
    vaSignature: Type = TStruct.empty,
    globalSignature: Type = TStruct.empty,
    genotypeSignature: Type = TGenotype,
    wasSplit: Boolean = false,
    isLinearScale: Boolean = false,
    genomeReference: GenomeReference = GenomeReference.GRCh37): VSMFileMetadata = {
    VSMFileMetadata(
      VSMMetadata(saSignature, vaSignature, globalSignature, genotypeSignature, wasSplit, isLinearScale, genomeReference = genomeReference),
      VSMLocalValue(globalAnnotation, sampleIds,
        if (sampleAnnotations == null)
          Annotation.emptyIndexedSeq(sampleIds.length)
        else
          sampleAnnotations))
  }
}

case class VSMFileMetadata(
  metadata: VSMMetadata,
  localValue: VSMLocalValue)

case class VSMMetadata(
  saSignature: Type = TStruct.empty,
  vaSignature: Type = TStruct.empty,
  globalSignature: Type = TStruct.empty,
  genotypeSignature: Type = TGenotype,
  wasSplit: Boolean = false,
  isLinearScale: Boolean = false,
  isGenericGenotype: Boolean = false,
  genomeReference: GenomeReference = GenomeReference.GRCh37)
