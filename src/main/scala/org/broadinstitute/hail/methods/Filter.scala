package org.broadinstitute.hail.methods

import org.apache.spark.SparkContext
import org.broadinstitute.hail.Utils
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
  type FilterGenotypeWithSA = (Array[AnnotationData] => ((Variant, AnnotationData) => ((Int, Sample, Genotype) => Boolean)))
  type FilterGenotypePostSA = (Variant, AnnotationData) => ((Int, Sample, Genotype) => Boolean)
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
    p = Some(Utils.eval[T](t))
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
    "(v: org.broadinstitute.hail.variant.Variant, \n" +
    "__va: org.broadinstitute.hail.annotations.AnnotationData) => { \n" +
      "import org.broadinstitute.hail.methods.FilterUtils._; \n" +
      signatures(vas, "__va") +
      instantiate("va", "__va") +
      cond + " }: Boolean"}) {
  def apply(v: Variant, va: AnnotationData): Boolean = eval()(v, va)
}

class FilterSampleCondition(cond: String, sas: AnnotationSignatures)
  extends Evaluator[(Sample, AnnotationData) => Boolean](
    "(s: org.broadinstitute.hail.variant.Sample, \n" +
    "__sa: org.broadinstitute.hail.annotations.AnnotationData) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; " +
      signatures(sas, "__sa") +
      instantiate("sa", "__sa") +
      cond + " }: Boolean") {
  def apply(s: Sample, sa: AnnotationData): Boolean = eval()(s, sa)
}

class FilterGenotypeCondition(cond: String, vas: AnnotationSignatures, sas: AnnotationSignatures,
  sad: Array[AnnotationData])
  extends EvaluatorWithTransformation[FilterGenotypeWithSA, FilterGenotypePostSA](
    {"(__sa: Array[org.broadinstitute.hail.annotations.AnnotationData]) => {\n" +
      "import org.broadinstitute.hail.methods.FilterUtils._\n" +
      signatures(sas, "__sa") +
      makeArray("__saArray", "__sa", "__sa") +
      "(v: org.broadinstitute.hail.variant.Variant, " +
      "__va: org.broadinstitute.hail.annotations.AnnotationData) => {\n" +
      signatures(vas, "__va") +
      instantiate("va", "__va") +
      "(__sIndex: Int, " +
      "s: org.broadinstitute.hail.variant.Sample, " +
      "g: org.broadinstitute.hail.variant.Genotype) => {\n" +
      "val sa = __saArray(__sIndex)\n" +
      cond + " }: Boolean}}"}, t => t(sad)) {
  def apply(sa: Array[AnnotationData])(v: Variant, va: AnnotationData)(sIndex: Int, s: Sample, g: Genotype): Boolean =
    eval()(v, va)(sIndex, s, g)
}
