package org.broadinstitute.hail.variant

object VariantMetadata {
  def apply(contigLength: Map[String, Int],
    sampleIds: Array[String]): VariantMetadata = VariantMetadata(contigLength, sampleIds, None)
  def apply(contigLength: Map[String, Int],
    sampleIds: Array[String],
    vcfHeader: Array[String]): VariantMetadata = VariantMetadata(contigLength, sampleIds, Some(vcfHeader))
}

case class VariantMetadata(val contigLength: Map[String, Int],
  val sampleIds: Array[String],
  val vcfHeader: Option[Array[String]]) {

  def nContigs: Int = contigLength.size
  def nSamples: Int = sampleIds.size
  def vcfVersion: Option[String] = vcfHeader
  .map(lines => {
    val versionRegex = """^##fileformat=(.*)$""".r
    lines(0) match {
      case versionRegex(version) => version
    }
  })
}
