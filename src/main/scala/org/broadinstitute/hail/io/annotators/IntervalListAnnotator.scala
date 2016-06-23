package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._

object IntervalListAnnotator {
  def apply(filename: String, hConf: hadoop.conf.Configuration): (GenomicIntervalSet, Option[(Type, Map[GenomicInterval, Annotation])]) = {
    // this annotator reads files in the UCSC BED spec defined here: https://genome.ucsc.edu/FAQ/FAQformat.html#format1

    val stringAnno = readLines(filename, hConf) { lines =>

      if (lines.isEmpty)
        fatal("empty interval file")

      val firstLine = lines.next()
      firstLine.value match {
        case GenomicIntervalSet.intervalRegex(contig, start_str, end_str) => false
        case line if line.split("""\s+""").length == 5 => true
        case _ => fatal("unsupported interval list format")
      }
    }

    if (stringAnno) {
      val (gis, m) = GenomicIntervalSet.readWithMap(filename, hConf)
      (gis, Some(TString, m))
    } else {
      (GenomicIntervalSet.read(filename, hConf), None)
    }
  }
}
