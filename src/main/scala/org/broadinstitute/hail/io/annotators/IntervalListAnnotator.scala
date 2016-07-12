package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.{Interval, IntervalTree}
import org.broadinstitute.hail.variant._

object IntervalListAnnotator {

  val intervalRegex = """([^:]*)[:\t](\d+)[\-\t](\d+)""".r

  def readWithMap(filename: String, hConf: hadoop.conf.Configuration): (IntervalTree[Locus], Map[Interval[Locus], Annotation]) = {
    readLines(filename, hConf) { s =>
      val m = s
        .filter(line => !line.value.isEmpty && line.value(0) != '@')
        .map(_.transform { line => line.value match {
          case str => str.split("\t") match {
            case Array(contig, start, end, direction, target) =>
              assert(direction == "+" || direction == "-")
              // interval list is 1-based, inclusive: [start, end]
              (Interval(Locus(contig, start.toInt),
                Locus(contig, end.toInt + 1)), target)
            case _ => fatal(
              """invalid interval format.  Acceptable formats:
                |  `chr:start-end'
                |  `chr  start  end' (tab-separated)
                |  `chr  start  end  strand  target' (tab-separated, strand is `+' or `-')
              """.stripMargin)
          }
        }
        })
        .toMap
        .force

      (IntervalTree[Locus](m.keys.toArray), m)
    }
  }

  def read(filename: String, hConf: hadoop.conf.Configuration): IntervalTree[Locus] = {
    readLines(filename, hConf) { s =>
      val intervals = s
        .filter(line => !line.value.isEmpty && line.value(0) != '@')
        .map(_.transform { line => line.value match {
          case intervalRegex(contig, start_str, end_str) =>
            // interval list is 1-based, inclusive: [start, end]
            Interval(Locus(contig, start_str.toInt),
              Locus(contig, end_str.toInt + 1))
          case str => str.split("\t") match {
            case Array(contig, start, end, direction, _) =>
              assert(direction == "+" || direction == "-")
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
        }
        })
        .toArray

      IntervalTree(intervals)
    }
  }

  def write(it: IntervalTree[Locus], filename: String, hConf: hadoop.conf.Configuration) {
    writeTextFile(filename, hConf) { fw =>
      it.foreach { i =>
        assert(i.start.contig == i.end.contig)
        // interval list is 1-based, inclusive: [start, end]
        fw.write(i.start.contig + ":" + i.start.position + "-" + (i.end.position - 1) + "\n")
      }
    }
  }

  def apply(filename: String, hConf: hadoop.conf.Configuration): (IntervalTree[Locus], Option[(Type, Map[Interval[Locus], Annotation])]) = {
    val stringAnno = readLines(filename, hConf) { lines =>

      if (lines.isEmpty)
        fatal("empty interval file")

      val firstLine = lines.next()
      firstLine.value match {
        case intervalRegex(contig, start_str, end_str) => false
        case line if line.split("""\s+""").length == 5 => true
        case _ => fatal("unsupported interval list format")
      }
    }

    if (stringAnno) {
      val (gis, m) = readWithMap(filename, hConf)
      (gis, Some(TString, m))
    } else {
      (read(filename, hConf), None)
    }
  }
}
