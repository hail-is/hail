package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row

object Annotation {

  def empty: Annotation = null

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = IndexedSeq.fill[Annotation](n)(null)

  def emptySignature: Signature = EmptySignature()

  def printAnnotation(a: Any, nSpace: Int = 0): String = {
    val spaces = " " * nSpace
    a match {
      case r: Row =>
        "Row:\n" +
          r.toSeq.zipWithIndex.map { case (elem, index) =>
            s"""$spaces[$index] ${printAnnotation(elem, nSpace + 4)}"""
          }
            .mkString("\n")
      case null => "NULL"
      case a => a.toString + ": " + a.getClass.getName
    }
  }

  def apply(args: Any*): Annotation = Row.fromSeq(args)
}
