package is.hail.io.annotators

import is.hail.expr._
import is.hail.utils.{Interval, IntervalTree, _}
import is.hail.variant._
import org.apache.hadoop

import scala.collection.mutable

object IntervalListAnnotator {

  val intervalRegex = """([^:]*)[:\t](\d+)[\-\t](\d+)""".r

  def readWithMap(filename: String, hConf: hadoop.conf.Configuration): (IntervalTree[Locus], Map[Interval[Locus], List[String]]) = {
    hConf.readLines(filename) { s =>
      val m = mutable.Map.empty[Interval[Locus], List[String]]
      s
        .filter(line => !line.value.isEmpty && line.value(0) != '@')
        .foreach {
          _.foreach { line =>
            val (k, v) = line.split("\t") match {
              case Array(contig, start, end, direction, target) =>
                if (!(direction == "+" || direction == "-"))
                  fatal(s"expect `+' or `-' in the `direction' field, but found $direction")
                // interval list is 1-based, inclusive: [start, end]
                (Interval(Locus(contig, start.toInt), Locus(contig, end.toInt + 1)), target)
              case _ => fatal(
                """invalid interval format.  Acceptable formats:
                  |  `chr:start-end'
                  |  `chr  start  end' (tab-separated)
                  |  `chr  start  end  strand  target' (tab-separated, strand is `+' or `-')""".
                  stripMargin)
            }
            m.updateValue(k, Nil, prev => v :: prev)
          }
        }

      (IntervalTree[Locus](m.keys.toArray), m.toMap)
    }
  }

  def read(filename: String, hConf: hadoop.conf.Configuration, prune: Boolean = false): IntervalTree[Locus] = {
    hConf.readLines(filename) { lines =>
      val intervals = lines
        .filter(line => !line.value.isEmpty && line.value(0) != '@')
        .map(_.map {
          case intervalRegex(contig, startStr, endStr) =>
            // interval list is 1-based, inclusive: [start, end]
            Interval(Locus(contig, startStr.toInt),
              Locus(contig, endStr.toInt + 1))
          case str => str.split("\t") match {
            case Array(contig, start, end, direction, _) =>
              if (!(direction == "+" || direction == "-"))
                fatal(s"expect `+' or `-' in the `direction' field, but found $direction")
              // interval list is 1-based, inclusive: [start, end]
              Interval(Locus(contig, start.toInt),
                Locus(contig, end.toInt + 1))
            case _ => fatal(
              """invalid interval format.  Acceptable formats:
                |  `chr:start-end'
                |  `chr  start  end' (tab-separated)
                |  `chr  start  end  strand  target' (tab-separated, strand is `+' or `-')
              """.stripMargin)
          }
        }.value)
        .toArray

      IntervalTree(intervals, prune = prune)
    }
  }

  def write(it: IntervalTree[Locus], filename: String, hConf: hadoop.conf.Configuration) {
    hConf.writeTextFile(filename) {
      fw =>
        it.foreach { i =>
          assert(i.start.contig == i.end.contig)
          // interval list is 1-based, inclusive: [start, end]
          fw.write(i.start.contig + ":" + i.start.position + "-" + (i.end.position - 1) + "\n")
        }
    }
  }

  def apply(filename: String, hConf: hadoop.conf.Configuration): (IntervalTree[Locus], Option[(Type, Map[Interval[Locus], List[String]])]) = {
    val stringAnno = hConf.readLines(filename) {
      lines =>

        if (lines.isEmpty)
          fatal("empty interval file")

        val firstLine = lines.next()
        firstLine.map {
          case intervalRegex(contig, start_str, end_str) => false
          case line if line.split("""\s+""").length == 5 => true
          case _ => fatal("unsupported interval list format")
        }.value
    }

    if (stringAnno) {
      val (gis, m) = readWithMap(filename, hConf)
      (gis, Some(TString, m))
    } else {
      (read(filename, hConf), None)
    }
  }
}
