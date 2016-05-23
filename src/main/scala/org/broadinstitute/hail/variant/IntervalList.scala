package org.broadinstitute.hail.variant

import java.io.File
import java.util.TreeMap
import org.broadinstitute.hail.annotations.Annotation

import scala.collection.mutable
import scala.io.Source
import scala.collection.JavaConverters._
import org.apache.hadoop
import org.broadinstitute.hail.Utils._

case class Interval(contig: String, start: Int, end: Int, identifier: Option[Annotation] = None)

object IntervalList {

  val intervalRegex = """([^:]*):(\d+)-(\d+)""".r

  def apply(intervals: Traversable[Interval]): IntervalList = {
    val m = mutable.Map[String, TreeMap[Int, (Int, Option[Annotation])]]()
    intervals.foreach { case Interval(contig, start, end, identifier) =>
      m.getOrElseUpdate(contig, new TreeMap[Int, (Int, Option[Annotation])]()).put(start, (end, identifier))
    }
    new IntervalList(m)
  }

  def read(filename: String,
    hConf: hadoop.conf.Configuration): IntervalList = {

    readLines(filename, hConf) { s =>
      IntervalList(
        s.filter(line => !line.value.isEmpty && line.value(0) != '@')
          .map(_.transform { l =>
            l.value match {
              case intervalRegex(contig, start_str, end_str) =>
                Interval(contig, start_str.toInt, end_str.toInt)
              case line => line.split("\t") match {
                case Array(contig, start, end, direction, target) =>
                  // FIXME proper input error handling
                  assert(direction == "+" || direction == "-")
                  Interval(contig, start.toInt, end.toInt)
                case _ => fatal("invalid interval format" +
                  "\n  expected: `chr:start-end' or `chr  start  end  strand  target' (tab-separated)")
              }
            }
          })
          .toTraversable)
    }
  }
}

class IntervalList(private val m: mutable.Map[String, TreeMap[Int, (Int, Option[Annotation])]]) extends Serializable {
  def contains(p: (String, Int)): Boolean = {
    val (contig, pos) = p
    m.get(contig) match {
      case Some(t) =>
        val entry = t.floorEntry(pos)
        if (entry != null)
          pos <= entry.getValue()._1
        else
          false
      case None => false
    }
  }

  def query(p: (String, Int)): Option[Annotation] = {
    val (contig, pos) = p
    m.get(contig) match {
      case Some(t) =>
        val entry = t.floorEntry(pos)
        if (entry != null)
          if (pos <= entry.getValue._1)
            entry.getValue._2
          else
            None
        else
          None
      case None => None
    }
  }

  def write(filename: String, hConf: hadoop.conf.Configuration) {
    writeTextFile(filename, hConf) { fw =>
      for ((contig, t) <- m;
           entry <- t.entrySet().asScala)
        fw.write(contig + ":" + entry.getKey() + "-" + entry.getValue()._1 + "\n")
    }
  }

  override def equals(that: Any) = that match {
    case ilist: IntervalList => m == ilist.m
    case _ => false
  }
}
