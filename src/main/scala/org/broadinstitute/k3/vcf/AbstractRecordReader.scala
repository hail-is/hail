package org.broadinstitute.k3.vcf

import org.broadinstitute.k3.variant._

abstract class AbstractRecordReader {
  def readRecord(line: String): Option[(Variant, Iterator[Genotype])]
}

abstract class AbstractRecordReaderBuilder extends Serializable {
  def result(vcfHeaderLines: Array[String]): AbstractRecordReader
}
