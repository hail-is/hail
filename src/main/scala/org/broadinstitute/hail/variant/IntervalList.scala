package org.broadinstitute.hail.variant

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.check.Gen

import scala.math.max

case class GenomicIndex(contig: String, position: Int)

case class GenomicInterval(contig: String, start: Int, end: Int) {
  require(start <= end)

  def interval: Interval = Interval(start, end)
}

case class Interval(start: Int, end: Int) {
  require(start <= end)

  def contains(position: Int): Boolean = position >= start && position <= end
}

object Interval {
  def gen(min: Int, max: Int): Gen[Interval] = Gen.zip(Gen.choose(min, max), Gen.choose(min, max))
    .filter { case (x, y) => x <= y }
    .map { case (x, y) => Interval(x, y) }
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


case class IntervalTreeNode(i: Interval,
  left: Option[IntervalTreeNode],
  right: Option[IntervalTreeNode],
  maximum: Int) {

  def contains(position: Int): Boolean = {
    if (i.contains(position))
      true
    else if (position < i.start) {
      //      println("going left")
      left.exists(_.contains(position))
    }
    else {
      val rt = if (position <= maximum)
        right.exists(_.contains(position))
      else false
      val lft = if (left.exists(node => node.maximum >= position))
        left.exists(_.contains(position))
      else false
      rt || lft
    }
  }

  def query(position: Int): List[Interval] = {
    val lft = if (left.exists(_.maximum >= position))
      left.map(_.query(position)).getOrElse(Nil)
    else Nil
    val rt = if (position >= i.start && position <= maximum)
      right.map(_.query(position)).getOrElse(Nil)
    else Nil
    val l = if (i.contains(position))
      i :: lft ::: rt
    else lft ::: rt
    l
  }

  def intervals: List[Interval] = i :: left.map(_.intervals).getOrElse(Nil) ::: right.map(_.intervals).getOrElse(Nil)
}

case class IntervalSet(root: IntervalTreeNode) extends Serializable {
  def contains(position: Int): Boolean = root.contains(position)

  def query(position: Int): Set[Interval] = root.query(position).toSet

  def intervals: Set[Interval] = root.intervals.toSet
}

object IntervalSet {
  def apply(intervals: Array[Interval]): IntervalSet =
    new IntervalSet(fromSorted(intervals.sortBy(_.start), 0, intervals.length - 1).get)

  def fromSorted(intervals: Array[Interval], start: Int, end: Int): Option[IntervalTreeNode] = {
    if (start > end)
      None
    else {
      val mid = (start + end) / 2
      val i = intervals(mid)
      val lft = fromSorted(intervals, start, mid - 1)
      val rt = fromSorted(intervals, mid + 1, end)
      Some(IntervalTreeNode(i, lft, rt,
        max(i.end, max(lft.map(_.maximum).getOrElse(-1), rt.map(_.maximum).getOrElse(-1)))))
    }
  }

  def gen: Gen[IntervalSet] = {
    Gen.buildableOf[Array[Interval], Interval](Interval.gen(0, 100))
      .filter(_.nonEmpty)
      .map(intervals => IntervalSet(intervals))
  }

  def genQueries(t: IntervalSet, n: Int = 100): Gen[Array[Int]] = {
    val intervals = t.intervals
    val maximum = intervals.map(_.end).max
    val minimum = intervals.map(_.start).min
    val range = (maximum - minimum) / 10
    Gen.parameterized[Array[Int]] { p =>
      val rng = p.rng
      Gen.const((0 until n)
        .map(x => rng.nextInt(minimum - range, maximum + range))
        .toArray)
    }
  }
}
