package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.{Interval, IntervalList}

object BedAnnotator {
  def apply(filename: String,
    hConf: hadoop.conf.Configuration): (IntervalList, Type) = {
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

      val (getString, signature) = if (next.length < 4)
        (false, TBoolean)
      else
        (true, TString)

      val intervalList = IntervalList(
        dataLines
          .filter(l => !l.value.isEmpty)
          .map { l => l.transform { line =>
            val arr = line.value.split("""\s+""")
            Interval(arr(0), arr(1).toInt + 1, arr(2).toInt, //transform BED 0-based coordinates to Hail/VCF 1-based coordinates
              if (getString)
                Some(arr(3))
              else
                Some(true))
          }
          }
          .toTraversable)

      (intervalList, signature)
    }
  }
}
