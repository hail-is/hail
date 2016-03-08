package org.broadinstitute.hail.variant

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

object VariantMetadata {

  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(Array.empty[(String, String)],
    sampleIds,
    Annotations.emptyIndexedSeq(sampleIds.length),
    TStruct.empty,
    TStruct.empty)

  def apply(filters: IndexedSeq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[Annotations], sas: Type, vas: Type): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: IndexedSeq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotations],
  sampleAnnotationSignatures: Type,
  variantAnnotationSignatures: Type,
  wasSplit: Boolean = false) {

  def nSamples: Int = sampleIds.length

  // FIXME casts to asInstanceOf
  def addSampleAnnotations(sas: Type, sa: IndexedSeq[Annotations]): VariantMetadata =
    copy(
      sampleAnnotationSignatures = sampleAnnotationSignatures.asInstanceOf[TStruct] ++ sas.asInstanceOf[TStruct],
      sampleAnnotations = sampleAnnotations.zip(sa).map { case (a1, a2) => a1 ++ a2 })

  def addVariantAnnotationSignatures(newSigs: Type): VariantMetadata =
    copy(
      variantAnnotationSignatures = variantAnnotationSignatures.asInstanceOf[TStruct] ++ newSigs.asInstanceOf[TStruct])

  def addVariantAnnotationSignatures(key: String, sig: Type): VariantMetadata =
    copy(
      variantAnnotationSignatures = variantAnnotationSignatures.asInstanceOf[TStruct] +(key, sig))

  def removeVariantAnnotationSignatures(key: String): VariantMetadata =
    copy(
      variantAnnotationSignatures = variantAnnotationSignatures.asInstanceOf[TStruct] - key)
}
