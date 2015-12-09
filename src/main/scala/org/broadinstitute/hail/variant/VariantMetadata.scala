package org.broadinstitute.hail.variant

import org.broadinstitute.hail.annotations._

object VariantMetadata {
  def apply(contigLength: Map[String, Int],
    sampleIds: Array[String]): VariantMetadata = new VariantMetadata(contigLength, sampleIds, None,
    EmptySampleAnnotations(sampleIds.length), EmptyAnnotationSignatures(), EmptyAnnotationSignatures())

  def apply(contigLength: Map[String, Int],
    sampleIds: Array[String],
    vcfHeader: Array[String]): VariantMetadata = new VariantMetadata(contigLength, sampleIds, Some(vcfHeader),
      EmptySampleAnnotations(sampleIds.length), EmptyAnnotationSignatures(), EmptyAnnotationSignatures())

  def apply(contigLength: Map[String, Int], sampleIds: Array[String], vcfHeader: Array[String],
    sa: IndexedSeq[AnnotationData], sas: AnnotationSignatures, vas: AnnotationSignatures): VariantMetadata = {
    new VariantMetadata(contigLength, sampleIds, Some(vcfHeader), sa, sas, vas)
  }
}

case class VariantMetadata(contigLength: Map[String, Int],
  sampleIds: IndexedSeq[String],
  vcfHeader: Option[IndexedSeq[String]],
  sampleAnnotations: IndexedSeq[AnnotationData],
  sampleAnnotationSignatures: AnnotationSignatures,
  variantAnnotationSignatures: AnnotationSignatures) {

  def nContigs: Int = contigLength.size

  def nSamples: Int = sampleIds.length
}
