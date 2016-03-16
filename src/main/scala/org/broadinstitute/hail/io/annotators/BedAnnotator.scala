package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Signature, Annotation, SimpleSignature}
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.variant.{Interval, IntervalList, Variant}

object BedAnnotator {
  def apply(filename: String,
    hConf: hadoop.conf.Configuration): (IntervalList, Signature, String) = {
    // this annotator reads files in the UCSC BED spec defined here: https://genome.ucsc.edu/FAQ/FAQformat.html#format1

    readLines(filename, hConf) { lines =>

      val (header, remainder) = lines.span(line =>
        line.value.startsWith("browser") ||
          line.value.startsWith("track") ||
          line.value.matches("""^\w+=("[\w\d ]+"|\d+).*"""))

      val filt = header.filter(s => s.value.startsWith("track"))
        .toArray
      if (filt.length != 1)
        fatal("Invalid bed file: found 'track' in more than one header line")
      val nameR = """.*name="([\w\d\s]+)".*""".r
      val name = filt.head.value match {
        case nameR(str) => str
        case _ => fatal("Invalid bed file: could not find identifier 'name'")
      }

      fatalIf(remainder.isEmpty, "bed file contains interval lines")

      val dataLines = remainder.toArray
      fatalIf(dataLines.isEmpty, "bed file contains no data lines")
      val next = dataLines
        .head
        .value
        .split("""\s+""")
      val (getString, signature) = if (next.length < 4)
        (false, SimpleSignature(expr.TBoolean))
      else
        (true, SimpleSignature(expr.TString))

      val intervalList = IntervalList(
        dataLines
          .filter(l => !l.value.isEmpty)
          .map { l => l.transform(line => {
            val arr = line.value.split("""\s+""")
            if (getString)
              Interval(arr(0), arr(1).toInt, arr(2).toInt, Some(arr(3)))
            else
              Interval(arr(0), arr(1).toInt, arr(2).toInt, Some(true))
          })
          }
          .toTraversable)

      (intervalList, signature, name)
    }
  }
}
