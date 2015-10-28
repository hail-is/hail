package org.broadinstitute.hail.vcf

import org.broadinstitute.hail.variant._

abstract class AbstractRecordReader {
  def readRecord(line: String): Option[(Variant, Iterator[Genotype])]
}

abstract class AbstractRecordReaderBuilder extends Serializable {
  def result(vcfHeaderLines: Array[String]): AbstractRecordReader
}
