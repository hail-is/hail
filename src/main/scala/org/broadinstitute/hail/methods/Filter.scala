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
  override def toString: String = if (ot.isDefined) ot.get.toString else "NA" // ideal?
  def ===(that: FilterOption[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(_ == t)))
  def !==(that: FilterOption[T]): FilterOption[Boolean] = new FilterOption((this === that).ot.map(!_))
}

class FilterOptionBoolean(val ob: Option[Boolean]) extends AnyVal {
  def &&(that: FilterOptionBoolean): FilterOption[Boolean] = new FilterOption(this.ob.flatMap(b => that.ob.map(b && _)))
  def ||(that: FilterOptionBoolean): FilterOption[Boolean] = new FilterOption(this.ob.flatMap(b => that.ob.map(b || _)))
  def unary_!(): FilterOption[Boolean] = new FilterOption(this.ob.map(! _))
}

class FilterOptionString(val os: Option[String]) extends AnyVal {
  def toInt: FilterOption[Int] = new FilterOption(this.os.map(_.toInt))
  def toDouble: FilterOption[Double] = new FilterOption(this.os.map(_.toDouble))
  def +(that: FilterOptionString) = new FilterOption(this.os.flatMap(s => that.os.map(s + _)))
  def apply(i: Int): FilterOption[Char] = new FilterOption(this.os.map(_(i)))
}

class FilterOptionArray[T](val oa: Option[Array[T]]) extends AnyVal {
  def apply(i: Int): FilterOption[T] = new FilterOption(this.oa.map(_(i)))
  def size: FilterOption[Int] = new FilterOption(oa.map(_.size))
//  def ++(that: FilterArrayOption[T]): FilterOption[Array[T]] = new FilterOption(this.oa.flatMap(a => that.oa.map(a ++ _)))
}

// FIXME: replace with compare
class FilterOptionOrdered[T <: Ordered[T]](val ot: Option[T]) extends AnyVal {
  def >(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t > _)))
  def <(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t < _)))
  def >=(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t >= _)))
  def <=(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t <= _)))
}

class FilterOptionNumeric[T <: scala.math.Numeric[T]#Ops](val ot: Option[T]) extends AnyVal {
  def +(that: FilterOptionNumeric[T]): FilterOption[T] = new FilterOption(this.ot.flatMap(t => that.ot.map(t + _)))
  def -(that: FilterOptionNumeric[T]): FilterOption[T] = new FilterOption(this.ot.flatMap(t => that.ot.map(t - _)))
  def *(that: FilterOptionNumeric[T]): FilterOption[T] = new FilterOption(this.ot.flatMap(t => that.ot.map(t * _)))
  def unary_-(): FilterOption[T] = new FilterOption(this.ot.map(-_))
  def abs: FilterOption[T] = new FilterOption(this.ot.map(_.abs()))
  def signum: FilterOption[Int] = new FilterOption(this.ot.map(_.signum()))
  def toInt: FilterOption[Int] = new FilterOption(this.ot.map(_.toInt()))
  def toLong: FilterOption[Long] = new FilterOption(this.ot.map(_.toLong()))
  def toFloat: FilterOption[Float] = new FilterOption(this.ot.map(_.toFloat()))
  def toDouble: FilterOption[Double] = new FilterOption(this.ot.map(_.toDouble()))
}

object FilterUtils {
  implicit def toFilterString(s: String): FilterString = new FilterString(s)
  implicit def toFilterOption[T](t: T): FilterOption[T] = {println("converted to FO: " + t); new FilterOption(Some(t))}
  implicit def toFilterOptionBoolean(fo: FilterOption[Boolean]): FilterOptionBoolean = {println("converted to FOB: " + fo); new FilterOptionBoolean(fo.ot)}
  implicit def toFilterOptionString(fo: FilterOption[String]): FilterOptionString = {println("converted to FOS: " + fo); new FilterOptionString(fo.ot)}
  implicit def toFilterOptionArray[T](fo: FilterOption[Array[T]]): FilterOptionArray[T] = {println("converted to FOA: " + fo); new FilterOptionArray[T](fo.ot)}
  implicit def toFilterOptionOrdered[T <: Ordered[T]](fo: FilterOption[T]): FilterOptionOrdered[T] = {println("converted to FOO: " + fo); new FilterOptionOrdered[T](fo.ot)}
  implicit def toFilterOptionNumeric[T <: scala.math.Numeric[T]#Ops](fo: FilterOption[T]): FilterOptionNumeric[T] = {println("converted to FON: " + fo); new FilterOptionNumeric[T](fo.ot)}
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
  extends Evaluator[(Variant) => FilterOption[Boolean]](
    "(v: org.broadinstitute.hail.variant.Variant) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; " +
      cond + " }: Boolean") {

  def apply(v: Variant): Boolean = {
    val fo = eval()(v)
    if (fo.ot.isDefined) fo.ot.get else false
  }
}

class FilterSampleCondition(cond: String)
  extends Evaluator[(Sample) => FilterOption[Boolean]](
    "(s: org.broadinstitute.hail.variant.Sample) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; " +
      cond + " }: Boolean") {
  def apply(s: Sample): Boolean = {
    val fo = eval()(s)
    if (fo.ot.isDefined) fo.ot.get else false
  }
}

class FilterGenotypeCondition(cond: String)
  extends Evaluator[(Variant, Sample, Genotype) => FilterOption[Boolean]](
    "(v: org.broadinstitute.hail.variant.Variant, " +
      "s: org.broadinstitute.hail.variant.Sample, " +
      "g: org.broadinstitute.hail.variant.Genotype) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; " +
      cond + " }: Boolean") {
  def apply(v: Variant, s: Sample, g: Genotype): Boolean =
  {
    val fo = eval()(v, s, g)
    if (fo.ot.isDefined) fo.ot.get else false
  }
}
