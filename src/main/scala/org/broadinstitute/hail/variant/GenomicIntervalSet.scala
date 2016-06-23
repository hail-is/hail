package org.broadinstitute.hail.variant

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.utils.{Interval, IntervalSet}

case class GenomicIndex(contig: String, position: Int)

case class GenomicInterval(contig: String, start: Int, end: Int) {
  require(start <= end)

  def interval: Interval = Interval(start, end)
}

case class GenomicIntervalSet(trees: Map[String, IntervalSet]) extends Serializable {

  def contains(gi: GenomicIndex): Boolean = contains(gi.contig, gi.position)

  def contains(contig: String, pos: Int): Boolean = trees
    .get(contig)
    .exists(_.contains(pos))

  def query(contig: String, pos: Int): Set[GenomicInterval] = trees
    .get(contig)
    .map(_.query(pos).map(i => GenomicInterval(contig, i.start, i.end)))
    .getOrElse(Set.empty[GenomicInterval])

  def query(g: GenomicIndex): Set[GenomicInterval] = query(g.contig, g.position)

  def write(filename: String, hConf: hadoop.conf.Configuration) {
    writeTextFile(filename, hConf) { fw =>
      for (g <- trees.flatMap { case (contig, m) => m.intervals.map(i => GenomicInterval(contig, i.start, i.end)) })
        fw.write(g.contig + ":" + g.start + "-" + g.end + "\n")
    }
  }
}

object GenomicIntervalSet {

  def apply(intervals: Set[GenomicInterval]): GenomicIntervalSet = {
    GenomicIntervalSet(intervals.groupBy(_.contig)
      .mapValues(_.map(_.interval).toArray)
      .mapValues(IntervalSet(_))
      .force)

  }

  val intervalRegex = """([^:]*)[:\t](\d+)[\-\t](\d+)""".r

  def read(filename: String, hConf: hadoop.conf.Configuration): GenomicIntervalSet = {
    readLines(filename, hConf) { s =>

      val intervals = s
        .filter(line => !line.value.isEmpty && line.value(0) != '@')
        .map(_.transform { line => line.value match {
          case intervalRegex(contig, start_str, end_str) => GenomicInterval(contig, start_str.toInt, end_str.toInt)
          case str => str.split("\t") match {
            case Array(contig, start, end, direction, _) =>
              assert(direction == "+" || direction == "-")
              GenomicInterval(contig, start.toInt, end.toInt)
            case _ => fatal(
              """invalid interval format.  Acceptable formats:
                |  `chr:start-end'
                |  `chr  start  end' (tab-separated)
                |  `chr  start  end  strand  target' (tab-separated, strand is `+' or `-')
              """.stripMargin)
          }
        }
        })
        .toSet

      GenomicIntervalSet(intervals)
    }
  }

  def readWithMap(filename: String, hConf: hadoop.conf.Configuration): (GenomicIntervalSet, Map[GenomicInterval, Annotation]) = {
    readLines(filename, hConf) { s =>

      val m = s
        .filter(line => !line.value.isEmpty && line.value(0) != '@')
        .map(_.transform { line => line.value match {
          case str => str.split("\t") match {
            case Array(contig, start, end, direction, target) =>
              assert(direction == "+" || direction == "-")
              (GenomicInterval(contig, start.toInt, end.toInt), target)
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

      (GenomicIntervalSet(m.keySet), m)
    }
  }
}
