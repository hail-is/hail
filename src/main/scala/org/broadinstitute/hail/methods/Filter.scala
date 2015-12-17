package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils
import org.broadinstitute.hail.variant.{Sample, Genotype, Variant}
import scala.reflect.ClassTag
import scala.language.implicitConversions

class FilterString(val s: String) extends AnyVal {
  def ~(t: String): Boolean = s.r.findFirstIn(t).isDefined

  def !~(t: String): Boolean = !this.~(t)
}

object FilterUtils {
  implicit def toFilterString(s: String): FilterString = new FilterString(s)
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

class FilterVariantCondition(cond: String)
  extends Evaluator[(Variant) => Boolean](
    "(v: org.broadinstitute.hail.variant.Variant) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; " +
      cond + " }: Boolean") {
  def apply(v: Variant): Boolean = eval()(v)
}

class FilterSampleCondition(cond: String)
  extends Evaluator[(Sample) => Boolean](
    "(s: org.broadinstitute.hail.variant.Sample) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; " +
      cond + " }: Boolean") {
  def apply(s: Sample): Boolean = eval()(s)
}

class FilterGenotypeCondition(cond: String)
  extends Evaluator[(Variant, Sample, Genotype) => Boolean](
    "(v: org.broadinstitute.hail.variant.Variant, " +
      "s: org.broadinstitute.hail.variant.Sample, " +
      "g: org.broadinstitute.hail.variant.Genotype) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; " +
      cond + " }: Boolean") {
  def apply(v: Variant, s: Sample, g: Genotype): Boolean =
    eval()(v, s, g)
}
