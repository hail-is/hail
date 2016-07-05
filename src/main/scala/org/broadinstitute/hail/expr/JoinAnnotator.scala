package org.broadinstitute.hail.expr

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotation, _}

import scala.collection.mutable

trait JoinAnnotator {

  def splitWarning(leftSplit: Boolean, left: String, rightSplit: Boolean, right: String) {
    val msg =
      """Merge behavior may not be as expected, as all alternate alleles are
        |  part of the variant key.  See `annotatevariants' documentation for
        |  more information.""".stripMargin
    (leftSplit, rightSplit) match {
      case (true, true) =>
      case (false, false) => warn(
        s"""annotating an unsplit $left from an unsplit $right
            |  $msg""".stripMargin)
      case (true, false) => warn(
        s"""annotating a biallelic (split) $left from an unsplit $right
            |  $msg""".stripMargin)
      case (false, true) => warn(
        s"""annotating an unsplit $left from a biallelic (split) $right
            |  $msg""".stripMargin)
    }
  }

  def buildInserter(code: String, t: Type, ec: EvalContext, expectedHead: String): (Type, Inserter) = {
    val (parseTypes, fns) = Parser.parseAnnotationArgs(code, ec, expectedHead)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finaltype = parseTypes.foldLeft(t) { case (t, (ids, signature)) =>
      val (s, i) = t.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val f = (left: Annotation, right: Option[Annotation]) => {

      ec.setAll(left, right.orNull)

      val queries = fns.map(_ ())
      var newAnnotation = left
      queries.indices.foreach { i =>
        newAnnotation = inserters(i)(newAnnotation, queries(i))
      }
      newAnnotation
    }
    (finaltype, f)

  }
}
