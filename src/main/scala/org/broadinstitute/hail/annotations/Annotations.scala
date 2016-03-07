package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.Utils._

object Annotations {

  def empty: Annotation = null: Any

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = IndexedSeq.fill[Annotation](n)(null: Any)

  def emptySignature: Signature = EmptySignature()

  def printRow(r: Row, nSpace: Int = 0) {
    val spaces = (0 until nSpace).map(i => " ").foldRight("")(_ + _)
    r.toSeq.zipWithIndex.foreach { case (elem, index) =>
      elem match {
        case row: Row =>
          println(s"""$spaces[$index] ROW:""")
          printRow(row, nSpace + 2)
        case _ =>
          println(s"""$spaces[$index] $elem ${if (elem != null) elem.getClass.getName else ""}""")
      }
    }
  }
}
