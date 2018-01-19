package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.expr.types._

object VSMLocalValue {
  def apply(sampleIds: IndexedSeq[Annotation]): VSMLocalValue =
    VSMLocalValue(Annotation.empty,
      sampleIds,
      Annotation.emptyIndexedSeq(sampleIds.length))
}

case class VSMLocalValue(
  globalAnnotation: Annotation,
  sampleIds: IndexedSeq[Annotation],
  sampleAnnotations: IndexedSeq[Annotation]) {
  assert(sampleIds.length == sampleAnnotations.length)

  def nSamples: Int = sampleIds.length

  def dropSamples(): VSMLocalValue = VSMLocalValue(globalAnnotation,
    IndexedSeq.empty[Annotation],
    IndexedSeq.empty[Annotation])
}

object VSMFileMetadata {
  def apply(metadata: VSMMetadata, localValue: VSMLocalValue): VSMFileMetadata = VSMFileMetadata(metadata, localValue, None)

  def apply(sampleIds: IndexedSeq[String],
    sampleAnnotations: IndexedSeq[Annotation] = null,
    globalAnnotation: Annotation = Annotation.empty,
    sSignature: Type = TString(),
    saSignature: TStruct = TStruct.empty(),
    vSignature: Type = TVariant(GenomeReference.defaultReference),
    vaSignature: TStruct = TStruct.empty(),
    globalSignature: TStruct = TStruct.empty(),
    // FIXME require
    genotypeSignature: TStruct = Genotype.htsGenotypeType): VSMFileMetadata = {
    VSMFileMetadata(
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature),
      VSMLocalValue(globalAnnotation, sampleIds,
        if (sampleAnnotations == null)
          Annotation.emptyIndexedSeq(sampleIds.length)
        else
          sampleAnnotations))
  }
}

case class VSMFileMetadata(
  metadata: VSMMetadata,
  localValue: VSMLocalValue,
  partitionCounts: Option[Array[Long]]) {
}

case class VSMMetadata(
  sSignature: Type = TString(),
  saSignature: TStruct = TStruct.empty(),
  vSignature: Type = TVariant(GenomeReference.defaultReference),
  vaSignature: TStruct = TStruct.empty(),
  globalSignature: TStruct = TStruct.empty(),
  genotypeSignature: TStruct = Genotype.htsGenotypeType)
