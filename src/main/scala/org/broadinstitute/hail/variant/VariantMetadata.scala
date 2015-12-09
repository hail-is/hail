package org.broadinstitute.hail.variant

case class VariantMetadata(contigLength: Map[String, Int],
  sampleIds: IndexedSeq[String]) {
  def nContigs: Int = contigLength.size

  def nSamples: Int = sampleIds.length
}
