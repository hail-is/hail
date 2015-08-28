package org.broadinstitute.k3.methods

import java.io.File
import org.broadinstitute.k3.variant.IntervalList
import scala.io.Source

object LoadIntervalList {
  def updateValueOrElse[A, B](m: Map[A, B], a: A, f: (B) => B, default: => B): Map[A, B] = {
    m.updated(a, m.get(a) match {
      case None => f(default)
      case Some(x) => f(x)
    })
  }

  def apply(filename: String): IntervalList = {
    require(filename.endsWith(".interval_list"))

    val ilist = IntervalList()

    Source.fromFile(new File(filename))
    .getLines()
    .filter(line => !line.isEmpty && line(0) != '@')
    .foreach { line =>
      val fields: Array[String] = line.split("\t")
      assert(fields(3) == "+" || fields(3) == "-")
      val contig = fields(0)
      val start = fields(1).toInt
      val end = fields(2).toInt
      // val targetName = fields(4)
      ilist += (contig -> (start, end))
    }

    ilist
  }

}
