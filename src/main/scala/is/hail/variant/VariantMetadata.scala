package is.hail.variant

import is.hail.annotations._
import is.hail.expr._

object VariantMetadata {
  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(
    sampleIds,
    Annotation.emptyIndexedSeq(sampleIds.length),
    Annotation.empty,
    TStruct.empty,
    TStruct.empty,
    TStruct.empty)

  def apply(sampleIds: Array[String], wasSplit: Boolean): VariantMetadata = new VariantMetadata(
    sampleIds,
    Annotation.emptyIndexedSeq(sampleIds.length),
    Annotation.empty,
    TStruct.empty,
    TStruct.empty,
    TStruct.empty,
    wasSplit = wasSplit)

  def apply(
    sampleIds: Array[String],
    sa: IndexedSeq[Annotation],
    globalAnnotation: Annotation,
    sas: Type,
    vas: Type,
    globalSignature: Type): VariantMetadata = {
    new VariantMetadata(sampleIds, sa, globalAnnotation, sas, vas, globalSignature)
  }
}

case class VariantMetadata(
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotation],
  globalAnnotation: Annotation,
  saSignature: Type,
  vaSignature: Type,
  globalSignature: Type,
  genotypeSignature: Type = TGenotype,
  wasSplit: Boolean = false,
  isDosage: Boolean = false,
  isGenericGenotype: Boolean = false) {

  def nSamples: Int = sampleIds.length
}
