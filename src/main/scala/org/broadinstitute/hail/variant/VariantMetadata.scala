package org.broadinstitute.hail.variant

import org.broadinstitute.hail.annotations._

object VariantMetadata {
  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(Seq.empty[(String, String)],
    sampleIds, Annotations.emptyOfArrayString(sampleIds.length), Annotations.emptyOfSignature(),
    Annotations.emptyOfSignature())

  def apply(filters: Seq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[AnnotationData], sas: AnnotationSignatures, vas: AnnotationSignatures): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: Seq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[AnnotationData],
  sampleAnnotationSignatures: AnnotationSignatures,
  variantAnnotationSignatures: AnnotationSignatures) {

  def nSamples: Int = sampleIds.length

  def addSampleAnnotations(sas: AnnotationSignatures, sa: IndexedSeq[AnnotationData]): VariantMetadata = {
    copy(
      sampleAnnotationSignatures = sampleAnnotationSignatures ++ sas,
      sampleAnnotations = sampleAnnotations.zip(sa).map { case (a1, a2) => a1 ++ a2 }
    )
  }
}
