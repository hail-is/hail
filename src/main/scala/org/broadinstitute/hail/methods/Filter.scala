package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils
import org.broadinstitute.hail.variant.{Sample, Genotype, Variant}
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
  //def ++(that: FilterOptionArray[T]): FilterOption[Array[T]] = new FilterOption(this.oa.flatMap(a => that.oa.map(a ++ _)))
}

// tricky to replace with compare
class FilterOptionOrdered[T](val ot: Option[T])(implicit order: (T) => Ordered[T]) {
  def >(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t > _)))
  def <(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t < _)))
  def >=(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t >= _)))
  def <=(that: FilterOptionOrdered[T]): FilterOption[Boolean] = new FilterOption(this.ot.flatMap(t => that.ot.map(t <= _)))
}

class FilterOptionNumeric[T](val ot: Option[T])(implicit order: (T) => scala.math.Numeric[T]#Ops) {
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

class FilterOptionDouble(val od: Option[Double]) {
  def /(that: FilterOptionDouble): FilterOption[Double] = new FilterOption(this.od.flatMap(d => that.od.map(d / _)))
}

class FilterOptionFloat(val of: Option[Float]) {
  def /(that: FilterOptionFloat): FilterOption[Float] = new FilterOption(this.of.flatMap(f => that.of.map(f / _)))
  def /(that: FilterOptionDouble): FilterOption[Double] = new FilterOption(this.of.flatMap(f => that.od.map(f / _)))
}

class FilterOptionLong(val ol: Option[Long]) {
  def /(that: FilterOptionLong): FilterOption[Long] = new FilterOption(this.ol.flatMap(l => that.ol.map(l / _)))
  def /(that: FilterOptionFloat): FilterOption[Float] = new FilterOption(this.ol.flatMap(l => that.of.map(l / _)))
  def /(that: FilterOptionDouble): FilterOption[Double] = new FilterOption(this.ol.flatMap(l => that.od.map(l / _)))
}

class FilterOptionInt(val oi: Option[Int]) {
  def /(that: FilterOptionInt): FilterOption[Int] = new FilterOption(this.oi.flatMap(i => that.oi.map(i / _)))
  def /(that: FilterOptionLong): FilterOption[Long] = new FilterOption(this.oi.flatMap(i => that.ol.map(i / _)))
  def /(that: FilterOptionFloat): FilterOption[Float] = new FilterOption(this.oi.flatMap(i => that.of.map(i / _)))
  def /(that: FilterOptionDouble): FilterOption[Double] = new FilterOption(this.oi.flatMap(i => that.od.map(i / _)))
}

class FilterObject[T](val t: T) extends AnyVal {
  def ===(that: T) = t == that
}

object FilterUtils {
  implicit def toFilterString(s: String): FilterString = new FilterString(s)

  implicit def toFilterObject[T](t: T): FilterObject[T] = new FilterObject(t)

  implicit def toFilterOption[T](t: T): FilterOption[T] = new FilterOption(Some(t))

  implicit def toFilterOptionBoolean(fo: FilterOption[Boolean]): FilterOptionBoolean = new FilterOptionBoolean(fo.ot)
  implicit def toFilterOptionString(fo: FilterOption[String]): FilterOptionString = new FilterOptionString(fo.ot)
  implicit def toFilterOptionArray[T](fo: FilterOption[Array[T]]): FilterOptionArray[T] = new FilterOptionArray[T](fo.ot)

  implicit def toFilterOptionOrdered[T](fo: FilterOption[T])(implicit order: (T) => Ordered[T]): FilterOptionOrdered[T] = new FilterOptionOrdered[T](fo.ot)
  implicit def toFilterOptionOrdered[T](t: T)(implicit order: (T) => Ordered[T]): FilterOptionOrdered[T] = new FilterOptionOrdered[T](Some(t))

  implicit def toFilterOptionNumeric[T](fo: FilterOption[T])(implicit order: (T) => scala.math.Numeric[T]#Ops): FilterOptionNumeric[T] = new FilterOptionNumeric[T](fo.ot)
  implicit def toFilterOptionNumeric[T](t: T)(implicit order: (T) => scala.math.Numeric[T]#Ops): FilterOptionNumeric[T] = new FilterOptionNumeric[T](Some(t))

  implicit def toFilterOptionDouble(fo: FilterOption[Double]): FilterOptionDouble = new FilterOptionDouble(fo.ot)
  implicit def toFilterOptionFloat(fo: FilterOption[Float]): FilterOptionFloat = new FilterOptionFloat(fo.ot)
  implicit def toFilterOptionLong(fo: FilterOption[Long]): FilterOptionLong = new FilterOptionLong(fo.ot)
  implicit def toFilterOptionInt(fo: FilterOption[Int]): FilterOptionInt = new FilterOptionInt(fo.ot)
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
      "import org.broadinstitute.hail.methods.FilterUtils._; import org.broadinstitute.hail.methods._; " +
      cond + " }: org.broadinstitute.hail.methods.FilterOption[Boolean]") {
  def apply(v: Variant) = eval()(v)
}

class FilterSampleCondition(cond: String)
  extends Evaluator[(Sample) => FilterOption[Boolean]](
    "(s: org.broadinstitute.hail.variant.Sample) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._; import org.broadinstitute.hail.methods._; " +
      cond + " }: org.broadinstitute.hail.methods.FilterOption[Boolean]") {
  def apply(s: Sample) = eval()(s)
}

class FilterGenotypeCondition(cond: String)
  extends Evaluator[(Variant, Sample, Genotype) => FilterOption[Boolean]](
    "(v: org.broadinstitute.hail.variant.Variant, " +
      "s: org.broadinstitute.hail.variant.Sample, " +
      "g: org.broadinstitute.hail.variant.Genotype) => { " +
      "import org.broadinstitute.hail.methods.FilterUtils._;  import org.broadinstitute.hail.methods._; " +
      cond + " }: org.broadinstitute.hail.methods.FilterOption[Boolean]") {
  def apply(v: Variant, s: Sample, g: Genotype) = eval()(v, s, g)
}

//def toBoolean: Boolean = if (this.ob.isDefined) this.ob.get else false //FIXME: need to condition on keep or remove

