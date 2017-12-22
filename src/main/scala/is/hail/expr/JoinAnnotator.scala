package is.hail.expr

import is.hail.annotations._
import is.hail.expr.typ._
import is.hail.utils._

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
    val (paths, types, f) = Parser.parseAnnotationExprs(code, ec, Some(expectedHead))

    val inserterBuilder = new ArrayBuilder[Inserter]()
    val finalType = (paths, types).zipped.foldLeft(t) { case (t, (ids, signature)) =>
      val (s, i) = t.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val insF = (left: Annotation, right: Annotation) => {
      ec.setAll(left, right)

      var newAnnotation = left
      val queries = f()
      queries.indices.foreach { i =>
        newAnnotation = inserters(i)(newAnnotation, queries(i))
      }
      newAnnotation
    }

    (finalType, insF)
  }
}
