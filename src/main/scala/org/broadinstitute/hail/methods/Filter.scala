package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.methods.FilterUtils.{FilterGenotypePostSA, FilterGenotypeWithSA}
import org.broadinstitute.hail.variant._
import scala.reflect.ClassTag
import scala.language.implicitConversions

class FilterString(val s: String) extends AnyVal {
  def ~(t: String): Boolean = s.r.findFirstIn(t).isDefined

  def !~(t: String): Boolean = !this.~(t)
}

object ConvertibleString {
  val someRegex = """Some\(([0-9\.]+)\)""".r
}

class ConvertibleString(val s: String) extends AnyVal {
  def toArrayInt: Array[Int] = s.split(",").map(i => i.toInt)

  def toArrayDouble: Array[Double] = s.split(",").map(i => i.toDouble)

  def toSetString: Set[String] = s.split(",").toSet

  def toStupidAnnotation: Array[Array[String]] = s.split(",").map(_.split("|").map(_.trim))

  def toOptionInt: Option[Int] = s match {
    case ConvertibleString.someRegex(i) => Some(i.toInt)
    case "None" => None
  }

  def toOptionDouble: Option[Double] = s match {
    case ConvertibleString.someRegex(i) => Some(i.toDouble)
    case "None" => None
  }
}

object FilterUtils {
  type FilterGenotypeWithSA = ((IndexedSeq[AnnotationData], IndexedSeq[String]) =>
    ((Variant, AnnotationData) => ((Int, Genotype) => Boolean)))
  type FilterGenotypePostSA = (Variant, AnnotationData) => ((Int, Genotype) => Boolean)

  implicit def toFilterString(s: String): FilterString = new FilterString(s)

  implicit def toConvertibleString(s: String): ConvertibleString = new ConvertibleString(s)

  //  def test(): (Variant, Annotations[String]) => Boolean = {
  //    throw new UnsupportedOperationException
  //  }
}

class EvaluatorWithTransformation[T, S](t: String, f: T => S)(implicit tct: ClassTag[T]) extends Serializable {
  @transient var p: Option[S] = None

  def typeCheck() {
    require(p.isEmpty)
    p = Some(f(Utils.eval[T](t)))
  }

  def eval(): S = p match {
    case null | None =>
      val v = f(Utils.eval[T](t))
      p = Some(v)
      v
    case Some(v) => v
  }
}

class Evaluator[T](t: String)(implicit tct: ClassTag[T])
  extends Serializable {
  @transient var p: Option[T] = None

  def typeCheck() {
    require(p.isEmpty)
    try {
      p = Some(Utils.eval[T](t))
    }
    catch {
      case e: scala.tools.reflect.ToolBoxError =>
        /* e.message looks like:
           reflective compilation has failed:

           ';' expected but '.' found. */
        fatal("parse error in condition: " + e.message.split("\n").last)
    }
  }

  def eval(): T = p match {
    case null | None =>
      val v = Utils.eval[T](t)
      p = Some(v)
      v
    case Some(v) => v
  }
}

class FilterVariantCondition(cond: String, vas: AnnotationSignatures)
  extends Evaluator[(Variant, AnnotationData) => Boolean]({
    s"""(v: org.broadinstitute.hail.variant.Variant,
        |  __va: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  ${signatures(vas, "__va")}
        |  ${instantiate("va", "__va")}
        |  $cond
        |}: Boolean
    """.stripMargin
  }) {
  def apply(v: Variant, va: AnnotationData): Boolean = eval()(v, va)
}

class FilterSampleCondition(cond: String, sas: AnnotationSignatures)
  extends Evaluator[(Sample, AnnotationData) => Boolean](
    s"""(s: org.broadinstitute.hail.variant.Sample,
        |  __sa: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  ${signatures(sas, "__sa")}
        |  ${instantiate("sa", "__sa")}
        |  $cond
        |}: Boolean
    """.stripMargin) {
  def apply(s: Sample, sa: AnnotationData): Boolean = eval()(s, sa)
}

class FilterGenotypeCondition(cond: String, vas: AnnotationSignatures, sas: AnnotationSignatures,
  sad: IndexedSeq[AnnotationData], ids: IndexedSeq[String])
  extends EvaluatorWithTransformation[FilterGenotypeWithSA, FilterGenotypePostSA](
        s"""(__sa: IndexedSeq[org.broadinstitute.hail.annotations.AnnotationData],
            |  __ids: IndexedSeq[String]) => {
            |  import org.broadinstitute.hail.methods.FilterUtils._
            |  ${signatures(sas, "__sa")}
            |  ${makeIndexedSeq("__saArray", "__sa", "__sa")}
            |  (v: org.broadinstitute.hail.variant.Variant,
            |    __va: org.broadinstitute.hail.annotations.AnnotationData) => {
            |    ${signatures(vas, "__va")}
            |    ${instantiate("va", "__va")}
            |    (__sIndex: Int,
            |      g: org.broadinstitute.hail.variant.Genotype) => {
            |        val sa = __saArray(__sIndex)
            |        val s = org.broadinstitute.hail.variant.Sample(__ids(__sIndex))
            |        $cond
            |      }: Boolean
            |   }
            | }
      """.stripMargin,
    t => t(sad, ids)) {
  def apply(v: Variant, va: AnnotationData)(sIndex: Int, g: Genotype): Boolean =
    eval()(v, va)(sIndex, g)
}
