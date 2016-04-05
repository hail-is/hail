package org.broadinstitute.hail.variant

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

object VariantMetadata {

  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(Array.empty[(String, String)],
    sampleIds,
    Annotation.emptyIndexedSeq(sampleIds.length),
    TEmpty,
    TEmpty)

  def apply(filters: IndexedSeq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[Annotation], sas: TypeWithSchema, vas: TypeWithSchema): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: IndexedSeq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotation],
  saSignature: TypeWithSchema,
  vaSignature: TypeWithSchema,
  wasSplit: Boolean = false) {

  def nSamples: Int = sampleIds.length
}
