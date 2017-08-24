package is.hail.io.annotators

import is.hail.HailContext
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.utils.{Interval, _}
import is.hail.variant._
import org.apache.spark.sql.Row


object IntervalList {

  val intervalRegex = """([^:]*)[:\t](\d+)[\-\t](\d+)""".r

  def read(hc: HailContext, filename: String): KeyTable = {
    val hasValue = hc.hadoopConf.readLines(filename) {
      lines =>
        val skipHeader = lines.filter(l => !l.value.isEmpty && l.value(0) != '@')

        if (skipHeader.isEmpty)
          fatal("empty interval file")

        val firstLine = skipHeader.next()
        firstLine.map {
          case intervalRegex(contig, start_str, end_str) => false
          case line if line.split("""\s+""").length == 5 => true
          case _ => fatal(
            """invalid interval format.  Acceptable formats:
              |  `chr:start-end'
              |  `chr  start  end' (tab-separated)
              |  `chr  start  end  strand  target' (tab-separated, strand is `+' or `-')""".stripMargin)
        }.value
    }

    val rg = ReferenceGenome.GRCh37

    val schema = if (hasValue)
      TStruct("interval" -> TInterval(rg), "target" -> TString)
    else
      TStruct("interval" -> TInterval(rg))

    val rdd = hc.sc.textFileLines(filename)
      .filter(l => !l.value.isEmpty && l.value(0) != '@')
      .map {
        _.map { line =>
          if (hasValue) {
            val split = line.split("\\s+")
            split match {
              case Array(contig, start, end, dir, target) =>
                val interval = Interval(Locus(contig, start.toInt), Locus(contig, end.toInt + 1))
                Row(interval, target)
              case arr => fatal(s"expected 5 fields, but found ${ arr.length }")
            }
          } else {
            line match {
              case intervalRegex(contig, startStr, endStr) =>
                Row(Interval(Locus(contig, startStr.toInt), Locus(contig, endStr.toInt + 1)))
              case _ => fatal("invalid interval")
            }
          }
        }.value
      }

    KeyTable(hc, rdd, schema, Array("interval"))
  }
}
