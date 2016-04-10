package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row

object Annotation {

  final val SAMPLE_HEAD = "sa"

  final val VARIANT_HEAD = "va"

  def empty: Annotation = null

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = IndexedSeq.fill[Annotation](n)(null)

  def printAnnotation(a: Any, nSpace: Int = 0): String = {
    val spaces = " " * nSpace
    a match {
      case null => "NULL"
      case r: Row =>
        "Row:\n" +
          r.toSeq.zipWithIndex.map { case (elem, index) =>
            s"""$spaces[$index] ${printAnnotation(elem, nSpace + 4)}: ${if (elem != null)
              elem.getClass.getCanonicalName
            else
              "Null"}"""
          }
            .mkString("\n")
      case a => a.toString + ": " + a.getClass.getName
    }
  }

  def apply(args: Any*): Annotation = Row.fromSeq(args)
}
