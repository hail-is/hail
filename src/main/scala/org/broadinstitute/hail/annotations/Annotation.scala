package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.broadinstitute.hail.expr._

import scala.collection.mutable

object Annotation {

  def empty: Annotation = null

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = IndexedSeq.fill[Annotation](n)(null)

  def printAnnotation(a: Any, nSpace: Int = 0): String = {
    val spaces = " " * nSpace
    a match {
      case null => "NULL"
      case r: Row =>
        "Row:\n" +
          r.toSeq.zipWithIndex.map { case (elem, index) =>
            s"""$spaces[$index] ${printAnnotation(elem, nSpace + 4)}"""
          }
            .mkString("\n")
      case a => a.toString + ": " + a.getClass.getCanonicalName
    }
  }

  def zipAnnotations(args: Array[Annotation]): Annotation = {
    if (args.forall(_.isInstanceOf[Row])) {
      val size = args.head.asInstanceOf[Row].size
      val rows = args.map(_.asInstanceOf[Row]).toArray
      val propagated = (0 until size).map { i => rows.map(_.get(i))}.toArray
      Row.fromSeq(propagated.map(arr => zipAnnotations(arr)))
  }
    else {
      args.toSet
        .take(5)
    }
  }

  def apply(args: Any*): Annotation = Row.fromSeq(args)
}
