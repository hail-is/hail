package is.hail.io.annotators

import is.hail.expr._
import is.hail.utils.{Interval, IntervalTree, _}
import is.hail.variant._
import org.apache.hadoop

import scala.collection.mutable

object BedAnnotator {
  def apply(filename: String,
    hConf: hadoop.conf.Configuration): (IntervalTree[Locus], Option[(Type, Map[Interval[Locus], List[String]])]) = {
    // this annotator reads files in the UCSC BED spec defined here: https://genome.ucsc.edu/FAQ/FAQformat.html#format1

    hConf.readLines(filename) { lines =>

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
        val m = mutable.Map.empty[Interval[Locus], List[String]]
        dataLines
          .filter(l => !l.value.isEmpty)
          .foreach {
            _.foreach { line =>
              val Array(chrom, strStart, strEnd, value) = line.split("""\s+""")
              // transform BED 0-based coordinates to Hail/VCF 1-based coordinates
              val interval = Interval(Locus(chrom, strStart.toInt + 1), Locus(chrom, strEnd.toInt + 1))
              m.updateValue(interval, Nil, prev => value :: prev)
            }
          }
        val t = IntervalTree(m.keys.toArray)
        (t, Some(TString, m.toMap))
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
