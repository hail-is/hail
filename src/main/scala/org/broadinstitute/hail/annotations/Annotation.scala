package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row

object Annotation {

  def empty: Annotation = null

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = IndexedSeq.fill[Annotation](n)(null)

  def emptySignature: Signature = EmptySignature()

  def printRow(r: Row, nSpace: Int = 0) {
    val spaces = " " * nSpace
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

  def apply(args: Any*): Annotation = Row.fromSeq(args)
}
