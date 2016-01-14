package org.broadinstitute.hail.variant

import org.broadinstitute.hail.annotations._

object VariantMetadata {
  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(Seq.empty[(String, String)],
    sampleIds, IndexedSeq.fill(sampleIds.length)(Annotations.empty()), Annotations.empty(),
    Annotations.empty())

  def apply(filters: Seq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[Annotations], sas: Annotations, vas: Annotations): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: Seq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotations],
  sampleAnnotationSignatures: Annotations,
  variantAnnotationSignatures: Annotations) {

  def nSamples: Int = sampleIds.length
}
