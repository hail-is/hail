package org.broadinstitute.k3.variant

import java.util.TreeMap
import scala.collection.mutable

object IntervalList {
  def apply(): IntervalList = {
    new IntervalList(mutable.Map.empty)
  }
}

// FIXME immutable?
// FIXME val
case class IntervalList(m: mutable.Map[String, TreeMap[Int, Int]]) {
  def +=(kv: (String, (Int, Int))) {
    val (contig, (start, end)) = kv
    m.getOrElseUpdate(contig, new TreeMap[Int, Int]()).put(start, end)
  }

  def contains(p: (String, Int)): Boolean = {
    val (contig, pos) = p
    m.get(contig) match {
      case Some(t) =>
        val entry = t.floorEntry(pos)
        if (entry != null)
          pos <= entry.getValue()
        else
          false
      case None => false
    }
  }
}
