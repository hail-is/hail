package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils
import org.broadinstitute.hail.variant.{Sample, Genotype, Variant}
import org.scalacheck.Prop.True
import scala.reflect.ClassTag
import scala.language.implicitConversions

class FilterString(val s: String) extends AnyVal {
  def ~(t: String): Boolean = s.r.findFirstIn(t).isDefined
  def !~(t: String): Boolean = !this.~(t)
}

class FilterOption[T](val ot: Option[T]) extends AnyVal {
  override def toString: String = if (ot.isDefined) ot.get.toString else "NA"
  def ===(that: FilterOption[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(_ == t)))
  def !==(that: FilterOption[T]): FilterOption[Boolean] = new FilterOption((this === that).ot.map(!_))
}

class FilterBooleanOption(val ob: Option[Boolean]) extends AnyVal {
  def &&(that: FilterBooleanOption): FilterOption[Boolean] = new FilterOption(this.ob.flatMap(b => that.ob.map(b && _)))
  def ||(that: FilterBooleanOption): FilterOption[Boolean] = new FilterOption(this.ob.flatMap(b => that.ob.map(b || _)))
  def unary_!(): FilterOption[Boolean] = new FilterOption(this.ob.map(! _))
}

class FilterStringOption(val os: Option[String]) extends AnyVal {
  def toInt: FilterOption[Int] = new FilterOption(this.os.map(_.toInt))
  def toDouble: FilterOption[Double] = new FilterOption(this.os.map(_.toDouble))
  def +(that: FilterStringOption) = new FilterOption(this.os.flatMap(s => that.os.map(s + _)))
  def apply(i: Int): FilterOption[Char] = new FilterOption(this.os.map(_(i)))
}

class FilterArrayOption[T](val oa: Option[Array[T]]) extends AnyVal {
  def apply(i: Int): FilterOption[T] = new FilterOption(this.oa.map(_(i)))
  def size: FilterOption[Int] = new FilterOption(oa.map(_.size))
//  def ++(that: FilterArrayOption[T]): FilterOption[Array[T]] = new FilterOption(this.oa.flatMap(a => that.oa.map(a ++ _)))
}

// FIXME: replace with compare
class FilterOrderedOption[T <: Ordered[T]](val ot: Option[T]) extends AnyVal {
  def >(that: FilterOrderedOption[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t > _)))
  def <(that: FilterOrderedOption[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t < _)))
  def >=(that: FilterOrderedOption[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t >= _)))
  def <=(that: FilterOrderedOption[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t <= _)))
}

class FilterNumericOption[T <: scala.math.Numeric[T]#Ops](val ot: Option[T]) extends AnyVal {
  def +(that: FilterNumericOption[T]): FilterOption[T] = new FilterOption(this.ot.flatMap(t => that.ot.map(t + _)))
  def -(that: FilterNumericOption[T]): FilterOption[T] = new FilterOption(this.ot.flatMap(t => that.ot.map(t - _)))
  def *(that: FilterNumericOption[T]): FilterOption[T] = new FilterOption(this.ot.flatMap(t => that.ot.map(t * _)))
  def unary_-(): FilterOption[T] = new FilterOption(this.ot.map(-_))
  def abs: FilterOption[T] = new FilterOption(this.ot.map(_.abs()))
  def signum: FilterOption[Int] = new FilterOption(this.ot.map(_.signum()))
  def toInt: FilterOption[Int] = new FilterOption(this.ot.map(_.toInt()))
  def toLong: FilterOption[Long] = new FilterOption(this.ot.map(_.toLong()))
  def toFloat: FilterOption[Float] = new FilterOption(this.ot.map(_.toFloat()))
  def toDouble: FilterOption[Double] = new FilterOption(this.ot.map(_.toDouble()))
}


object FilterUtils {
  implicit def toFilterOption[T](t: T): FilterOption[T] = {println("converted to FO: " + t); new FilterOption(Some(t))}
  implicit def toFilterBooleanOption(fo: FilterOption[Boolean]): FilterBooleanOption = {println("converted to FOB: " + fo); new FilterBooleanOption(fo.ot)}
  implicit def toFilterStringOption(fo: FilterOption[String]): FilterStringOption = {println("converted to FOS: " + fo); new FilterStringOption(fo.ot)}
  implicit def toFilterArrayOption[T](fo: FilterOption[Array[T]]): FilterArrayOption[T] = {println("converted to FOA: " + fo); new FilterArrayOption[T](fo.ot)}
  implicit def toFilterOrderedOption[T <: Ordered[T]](fo: FilterOption[T]): FilterOrderedOption[T] = {println("converted to FOO: " + fo); new FilterOrderedOption[T](fo.ot)}
  implicit def toFilterNumericOption[T <: scala.math.Numeric[T]#Ops](fo: FilterOption[T]): FilterNumericOption[T] = {println("converted to FON: " + fo); new FilterNumericOption[T](fo.ot)}
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
