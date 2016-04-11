package org.broadinstitute.hail.variant

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

object VariantMetadata {

  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(
    sampleIds,
    Annotation.emptyIndexedSeq(sampleIds.length),
    TEmpty,
    TEmpty)

  def apply(
    sampleIds: Array[String],
    sa: IndexedSeq[Annotation],
    sas: Type,
    vas: Type): VariantMetadata = {
    new VariantMetadata(sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotation],
  saSignature: Type,
  vaSignature: Type,
  wasSplit: Boolean = false) {

  def nSamples: Int = sampleIds.length
}
