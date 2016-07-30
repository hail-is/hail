package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.{Interval, IntervalTree}
import org.broadinstitute.hail.variant._

object BedAnnotator {
  def apply(filename: String,
    hConf: hadoop.conf.Configuration): (IntervalTree[Locus], Option[(Type, Map[Interval[Locus], Annotation])]) = {
    // this annotator reads files in the UCSC BED spec defined here: https://genome.ucsc.edu/FAQ/FAQformat.html#format1

    readLines(filename, hConf) { lines =>

      val (header, remainder) = lines.span(line =>
        line.value.startsWith("browser") ||
          line.value.startsWith("track") ||
          line.value.matches("""^\w+=("[\w\d ]+"|\d+).*"""))

      if (remainder.isEmpty)
        fatal("bed file contains no interval lines")

      val dataLines = remainder.toArray
      if (dataLines.isEmpty)
        fatal("bed file contains no data lines")
      val next = dataLines
        .head
        .value
        .split("""\s+""")

      val getString = next.length >= 4

      if (getString) {
        val m = dataLines
          .filter(l => !l.value.isEmpty)
          .map(l => l.map { line =>
            val Array(chrom, strStart, strEnd, value) = line.split("""\s+""")
            // transform BED 0-based coordinates to Hail/VCF 1-based coordinates
            (Interval(Locus(chrom, strStart.toInt + 1),
              Locus(chrom, strEnd.toInt + 1)),
              value)
          }.value)
          .toMap
        val t = IntervalTree(m.keys.toArray)
        (t, Some(TString, m))
      } else {
        val t = IntervalTree(dataLines
          .filter(l => !l.value.isEmpty)
          .map(l => l.map { line =>
            val Array(chrom, strStart, strEnd) = line.split("""\s+""")
            // transform BED 0-based coordinates to Hail/VCF 1-based coordinates
            Interval(Locus(chrom, strStart.toInt + 1),
              Locus(chrom, strEnd.toInt + 1))
          }.value))
        (t, None)
      }
    }
  }
}
