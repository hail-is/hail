package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.methods.FilterUtils.{FilterGenotypePostSA, FilterGenotypeWithSA}
import org.broadinstitute.hail.variant.{VariantMetadata, Sample, Genotype, Variant}
import scala.language.implicitConversions
import scala.reflect.runtime.universe._
import scala.reflect.runtime.currentMirror
import scala.tools.reflect.ToolBox

class FilterString(val s: String) extends AnyVal {
  def ~(t: String): Boolean = s.r.findFirstIn(t).isDefined

  def !~(t: String): Boolean = !this.~(t)
}

class AnnotationValueString(val s: String) extends AnyVal {
  def toArrayInt: Array[Int] = s.split(",").map(i => i.toInt)

  def toArrayDouble: Array[Double] = s.split(",").map(i => i.toDouble)

  def toSetString: Set[String] = s.split(",").toSet
}

object FilterOption {
  def apply[T](x: T): FilterOption[T] = new FilterOption[T](Some(x))

  def apply[T, S, V](a: Option[T], b: Option[S], f: (T, S) => V): FilterOption[V] =
    new FilterOption[V](a.flatMap(ax => b.map(bx => f(ax, bx))))

  def empty: FilterOption[Nothing] = new FilterOption[Nothing](None)
}

class FilterOption[+T](val o: Option[T]) extends AnyVal {
  override def toString: String = if (o.isDefined) o.get.toString else "NA"

  def fEq(that: Any): FilterOption[Boolean] =
    new FilterOption(o.flatMap(t =>
      that match {
        case s: FilterOption[_] => s.o.map(t == _)
        case _ => Some(t == that)
      }))

  def fNotEq(that: Any): FilterOption[Boolean] = new FilterOption((this fEq that).o.map(!_))
}

class FilterOptionBoolean(val o: Option[Boolean]) extends AnyVal {
  def fAnd(that: FilterOptionBoolean): FilterOption[Boolean] = FilterOption[Boolean, Boolean, Boolean](o, that.o, _ && _)

  def fOr(that: FilterOptionBoolean): FilterOption[Boolean] = FilterOption[Boolean, Boolean, Boolean](o, that.o, _ || _)

  def unary_!(): FilterOption[Boolean] = new FilterOption(o.map(!_))
}

class FilterOptionString(val o: Option[String]) extends AnyVal {
  def fApply(i: Int): FilterOption[Char] = new FilterOption(o.map(_ (i)))

  def length: FilterOption[Int] = new FilterOption(o.map(_.length))

  def fConcat(that: FilterOptionString) = FilterOption[String, String, String](o, that.o, _ + _)

  def fToInt: FilterOption[Int] = new FilterOption(o.map(_.toInt))
  def fToLong: FilterOption[Long] = new FilterOption(o.map(_.toLong))
  def fToFloat: FilterOption[Float] = new FilterOption(o.map(_.toFloat))
  def fToDouble: FilterOption[Double] = new FilterOption(o.map(_.toDouble))
}

class FilterOptionArray[T](val o: Option[Array[T]]) extends AnyVal {
  def fApply(i: Int): FilterOption[T] = new FilterOption(o.map(_ (i)))

  def fContains(fo: FilterOption[T]): FilterOption[Boolean] = FilterOption[Array[T], T, Boolean](o, fo.o, _.contains(_))
  def fContains(t: T): FilterOption[Boolean] = new FilterOption(o.map(_.contains(t)))

  def size: FilterOption[Int] = new FilterOption(o.map(_.length))

  def length: FilterOption[Int] = new FilterOption(o.map(_.length))

  //Fixme: error related to conversion to ArraySeq
  //def fConcat(that: FilterOptionArray[T]): FilterOption[Array[T]] = new FilterOption(oa.flatMap(a => that.oa.map(a ++ _)))
}

class FilterOptionSet[T](val o: Option[Set[T]]) extends AnyVal {

  def fContains(fo: FilterOption[T]): FilterOption[Boolean] = FilterOption[Set[T], T, Boolean](o, fo.o, _.contains(_))
  def fContains(t: T): FilterOption[Boolean] = new FilterOption(o.map(_.contains(t)))

  def size: FilterOption[Int] = new FilterOption(o.map(_.size))
}

class FilterOptionInt(val o: Option[Int]) {
  def fPlus(that: FilterOptionInt): FilterOption[Int] = FilterOption[Int, Int, Int](o, that.o, _ + _)
  def fPlus(that: FilterOptionLong): FilterOption[Long] = FilterOption[Int, Long, Long](o, that.o, _ + _)
  def fPlus(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Int, Float, Float](o, that.o, _ + _)
  def fPlus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Int, Double, Double](o, that.o, _ + _)
  def fPlus(that: Int): FilterOption[Int] = new FilterOption(o.map(_ + that))
  def fPlus(that: Long): FilterOption[Long] = new FilterOption(o.map(_ + that))
  def fPlus(that: Float): FilterOption[Float] = new FilterOption(o.map(_ + that))
  def fPlus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ + that))

  def fMinus(that: FilterOptionInt): FilterOption[Int] = FilterOption[Int, Int, Int](o, that.o, _ - _)
  def fMinus(that: FilterOptionLong): FilterOption[Long] = FilterOption[Int, Long, Long](o, that.o, _ - _)
  def fMinus(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Int, Float, Float](o, that.o, _ - _)
  def fMinus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Int, Double, Double](o, that.o, _ - _)
  def fMinus(that: Int): FilterOption[Int] = new FilterOption(o.map(_ - that))
  def fMinus(that: Long): FilterOption[Long] = new FilterOption(o.map(_ - that))
  def fMinus(that: Float): FilterOption[Float] = new FilterOption(o.map(_ - that))
  def fMinus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ - that))

  def fTimes(that: FilterOptionInt): FilterOption[Int] = FilterOption[Int, Int, Int](o, that.o, _ * _)
  def fTimes(that: FilterOptionLong): FilterOption[Long] = FilterOption[Int, Long, Long](o, that.o, _ * _)
  def fTimes(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Int, Float, Float](o, that.o, _ * _)
  def fTimes(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Int, Double, Double](o, that.o, _ * _)
  def fTimes(that: Int): FilterOption[Int] = new FilterOption(o.map(_ * that))
  def fTimes(that: Long): FilterOption[Long] = new FilterOption(o.map(_ * that))
  def fTimes(that: Float): FilterOption[Float] = new FilterOption(o.map(_ * that))
  def fTimes(that: Double): FilterOption[Double] = new FilterOption(o.map(_ * that))

  def fDiv(that: FilterOptionInt): FilterOption[Int] = FilterOption[Int, Int, Int](o, that.o, _ / _)
  def fDiv(that: FilterOptionLong): FilterOption[Long] = FilterOption[Int, Long, Long](o, that.o, _ / _)
  def fDiv(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Int, Float, Float](o, that.o, _ / _)
  def fDiv(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Int, Double, Double](o, that.o, _ / _)
  def fDiv(that: Int): FilterOption[Int] = new FilterOption(o.map(_ / that))
  def fDiv(that: Long): FilterOption[Long] = new FilterOption(o.map(_ / that))
  def fDiv(that: Float): FilterOption[Float] = new FilterOption(o.map(_ / that))
  def fDiv(that: Double): FilterOption[Double] = new FilterOption(o.map(_ / that))

  def fMod(that: FilterOptionInt): FilterOption[Int] = FilterOption[Int, Int, Int](o, that.o, _ % _)
  def fMod(that: FilterOptionLong): FilterOption[Long] = FilterOption[Int, Long, Long](o, that.o, _ % _)
  def fMod(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Int, Float, Float](o, that.o, _ % _)
  def fMod(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Int, Double, Double](o, that.o, _ % _)
  def fMod(that: Int): FilterOption[Int] = new FilterOption(o.map(_ % that))
  def fMod(that: Long): FilterOption[Long] = new FilterOption(o.map(_ % that))
  def fMod(that: Float): FilterOption[Float] = new FilterOption(o.map(_ % that))
  def fMod(that: Double): FilterOption[Double] = new FilterOption(o.map(_ % that))
  
  def unary_-(): FilterOption[Int] = new FilterOption(o.map(- _))
  def unary_+(): FilterOption[Int] = new FilterOption(o)

  def fAbs: FilterOption[Int] = new FilterOption(o.map(_.abs))

  def fSignum: FilterOption[Int] = new FilterOption(o.map(_.signum))

  def fMax(that: FilterOptionInt): FilterOption[Int] = FilterOption[Int, Int, Int](o, that.o, _ max _)
  def fMax(that: Int): FilterOption[Int] = new FilterOption(o.map(_ max that))

  def fMin(that: FilterOptionInt): FilterOption[Int] = FilterOption[Int, Int, Int](o, that.o, _ min _)
  def fMin(that: Int): FilterOption[Int] = new FilterOption(o.map(_ min that))

  def fToDouble: FilterOption[Double] = new FilterOption(o.map(_.toDouble))
  def fToFloat: FilterOption[Float] = new FilterOption(o.map(_.toFloat))
  def fToLong: FilterOption[Long] = new FilterOption(o.map(_.toLong))

  def fLt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Int, Int, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Int, Long, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Int, Float, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Int, Double, Boolean](o, that.o, _ < _)
  def fLt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ < that))

  def fGt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Int, Int, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Int, Long, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Int, Float, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Int, Double, Boolean](o, that.o, _ > _)
  def fGt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ > that))

  def fLe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Int, Int, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Int, Long, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Int, Float, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Int, Double, Boolean](o, that.o, _ <= _)
  def fLe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))

  def fGe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Int, Int, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Int, Long, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Int, Float, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Int, Double, Boolean](o, that.o, _ >= _)
  def fGe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
}

class FilterOptionLong(val o: Option[Long]) {
  def fPlus(that: FilterOptionInt): FilterOption[Long] = FilterOption[Long, Int, Long](o, that.o, _ + _)
  def fPlus(that: FilterOptionLong): FilterOption[Long] = FilterOption[Long, Long, Long](o, that.o, _ + _)
  def fPlus(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Long, Float, Float](o, that.o, _ + _)
  def fPlus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Long, Double, Double](o, that.o, _ + _)
  def fPlus(that: Int): FilterOption[Long] = new FilterOption(o.map(_ + that))
  def fPlus(that: Long): FilterOption[Long] = new FilterOption(o.map(_ + that))
  def fPlus(that: Float): FilterOption[Float] = new FilterOption(o.map(_ + that))
  def fPlus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ + that))

  def fMinus(that: FilterOptionInt): FilterOption[Long] = FilterOption[Long, Int, Long](o, that.o, _ - _)
  def fMinus(that: FilterOptionLong): FilterOption[Long] = FilterOption[Long, Long, Long](o, that.o, _ - _)
  def fMinus(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Long, Float, Float](o, that.o, _ - _)
  def fMinus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Long, Double, Double](o, that.o, _ - _)
  def fMinus(that: Int): FilterOption[Long] = new FilterOption(o.map(_ - that))
  def fMinus(that: Long): FilterOption[Long] = new FilterOption(o.map(_ - that))
  def fMinus(that: Float): FilterOption[Float] = new FilterOption(o.map(_ - that))
  def fMinus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ - that))

  def fTimes(that: FilterOptionInt): FilterOption[Long] = FilterOption[Long, Int, Long](o, that.o, _ * _)
  def fTimes(that: FilterOptionLong): FilterOption[Long] = FilterOption[Long, Long, Long](o, that.o, _ * _)
  def fTimes(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Long, Float, Float](o, that.o, _ * _)
  def fTimes(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Long, Double, Double](o, that.o, _ * _)
  def fTimes(that: Int): FilterOption[Long] = new FilterOption(o.map(_ * that))
  def fTimes(that: Long): FilterOption[Long] = new FilterOption(o.map(_ * that))
  def fTimes(that: Float): FilterOption[Float] = new FilterOption(o.map(_ * that))
  def fTimes(that: Double): FilterOption[Double] = new FilterOption(o.map(_ * that))

  def fDiv(that: FilterOptionInt): FilterOption[Long] = FilterOption[Long, Int, Long](o, that.o, _ / _)
  def fDiv(that: FilterOptionLong): FilterOption[Long] = FilterOption[Long, Long, Long](o, that.o, _ / _)
  def fDiv(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Long, Float, Float](o, that.o, _ / _)
  def fDiv(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Long, Double, Double](o, that.o, _ / _)
  def fDiv(that: Int): FilterOption[Long] = new FilterOption(o.map(_ / that))
  def fDiv(that: Long): FilterOption[Long] = new FilterOption(o.map(_ / that))
  def fDiv(that: Float): FilterOption[Float] = new FilterOption(o.map(_ / that))
  def fDiv(that: Double): FilterOption[Double] = new FilterOption(o.map(_ / that))

  def fMod(that: FilterOptionInt): FilterOption[Long] = FilterOption[Long, Int, Long](o, that.o, _ % _)
  def fMod(that: FilterOptionLong): FilterOption[Long] = FilterOption[Long, Long, Long](o, that.o, _ % _)
  def fMod(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Long, Float, Float](o, that.o, _ % _)
  def fMod(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Long, Double, Double](o, that.o, _ % _)
  def fMod(that: Int): FilterOption[Long] = new FilterOption(o.map(_ % that))
  def fMod(that: Long): FilterOption[Long] = new FilterOption(o.map(_ % that))
  def fMod(that: Float): FilterOption[Float] = new FilterOption(o.map(_ % that))
  def fMod(that: Double): FilterOption[Double] = new FilterOption(o.map(_ % that))

  def unary_-(): FilterOption[Long] = new FilterOption(o.map(- _))
  def unary_+(): FilterOption[Long] = new FilterOption(o)

  def fAbs: FilterOption[Long] = new FilterOption(o.map(_.abs))

  def fSignum: FilterOption[Long] = new FilterOption(o.map(_.signum))

  def fMax(that: FilterOptionLong): FilterOption[Long] = FilterOption[Long, Long, Long](o, that.o, _ max _)
  def fMax(that: Long): FilterOption[Long] = new FilterOption(o.map(_ max that))

  def fMin(that: FilterOptionLong): FilterOption[Long] = FilterOption[Long, Long, Long](o, that.o, _ min _)
  def fMin(that: Long): FilterOption[Long] = new FilterOption(o.map(_ min that))

  def fToDouble: FilterOption[Double] = new FilterOption(o.map(_.toDouble))
  def fToFloat: FilterOption[Float] = new FilterOption(o.map(_.toFloat))

  def fLt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Long, Int, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Long, Long, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Long, Float, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Long, Double, Boolean](o, that.o, _ < _)
  def fLt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ < that))

  def fGt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Long, Int, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Long, Long, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Long, Float, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Long, Double, Boolean](o, that.o, _ > _)
  def fGt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ > that))

  def fLe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Long, Int, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Long, Long, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Long, Float, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Long, Double, Boolean](o, that.o, _ <= _)
  def fLe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))

  def fGe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Long, Int, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Long, Long, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Long, Float, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Long, Double, Boolean](o, that.o, _ >= _)
  def fGe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
}

class FilterOptionFloat(val o: Option[Float]) {
  def fPlus(that: FilterOptionInt): FilterOption[Float] = FilterOption[Float, Int, Float](o, that.o, _ + _)
  def fPlus(that: FilterOptionLong): FilterOption[Float] = FilterOption[Float, Long, Float](o, that.o, _ + _)
  def fPlus(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Float, Float, Float](o, that.o, _ + _)
  def fPlus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Float, Double, Double](o, that.o, _ + _)
  def fPlus(that: Int): FilterOption[Float] = new FilterOption(o.map(_ + that))
  def fPlus(that: Long): FilterOption[Float] = new FilterOption(o.map(_ + that))
  def fPlus(that: Float): FilterOption[Float] = new FilterOption(o.map(_ + that))
  def fPlus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ + that))

  def fMinus(that: FilterOptionInt): FilterOption[Float] = FilterOption[Float, Int, Float](o, that.o, _ - _)
  def fMinus(that: FilterOptionLong): FilterOption[Float] = FilterOption[Float, Long, Float](o, that.o, _ - _)
  def fMinus(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Float, Float, Float](o, that.o, _ - _)
  def fMinus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Float, Double, Double](o, that.o, _ - _)
  def fMinus(that: Int): FilterOption[Float] = new FilterOption(o.map(_ - that))
  def fMinus(that: Long): FilterOption[Float] = new FilterOption(o.map(_ - that))
  def fMinus(that: Float): FilterOption[Float] = new FilterOption(o.map(_ - that))
  def fMinus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ - that))

  def fTimes(that: FilterOptionInt): FilterOption[Float] = FilterOption[Float, Int, Float](o, that.o, _ * _)
  def fTimes(that: FilterOptionLong): FilterOption[Float] = FilterOption[Float, Long, Float](o, that.o, _ * _)
  def fTimes(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Float, Float, Float](o, that.o, _ * _)
  def fTimes(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Float, Double, Double](o, that.o, _ * _)
  def fTimes(that: Int): FilterOption[Float] = new FilterOption(o.map(_ * that))
  def fTimes(that: Long): FilterOption[Float] = new FilterOption(o.map(_ * that))
  def fTimes(that: Float): FilterOption[Float] = new FilterOption(o.map(_ * that))
  def fTimes(that: Double): FilterOption[Double] = new FilterOption(o.map(_ * that))

  def fDiv(that: FilterOptionInt): FilterOption[Float] = FilterOption[Float, Int, Float](o, that.o, _ / _)
  def fDiv(that: FilterOptionLong): FilterOption[Float] = FilterOption[Float, Long, Float](o, that.o, _ / _)
  def fDiv(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Float, Float, Float](o, that.o, _ / _)
  def fDiv(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Float, Double, Double](o, that.o, _ / _)
  def fDiv(that: Int): FilterOption[Float] = new FilterOption(o.map(_ / that))
  def fDiv(that: Long): FilterOption[Float] = new FilterOption(o.map(_ / that))
  def fDiv(that: Float): FilterOption[Float] = new FilterOption(o.map(_ / that))
  def fDiv(that: Double): FilterOption[Double] = new FilterOption(o.map(_ / that))

  def fMod(that: FilterOptionInt): FilterOption[Float] = FilterOption[Float, Int, Float](o, that.o, _ % _)
  def fMod(that: FilterOptionLong): FilterOption[Float] = FilterOption[Float, Long, Float](o, that.o, _ % _)
  def fMod(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Float, Float, Float](o, that.o, _ % _)
  def fMod(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Float, Double, Double](o, that.o, _ % _)
  def fMod(that: Int): FilterOption[Float] = new FilterOption(o.map(_ % that))
  def fMod(that: Long): FilterOption[Float] = new FilterOption(o.map(_ % that))
  def fMod(that: Float): FilterOption[Float] = new FilterOption(o.map(_ % that))
  def fMod(that: Double): FilterOption[Double] = new FilterOption(o.map(_ % that))

  def unary_-(): FilterOption[Float] = new FilterOption(o.map(- _))
  def unary_+(): FilterOption[Float] = new FilterOption(o)

  def fAbs: FilterOption[Float] = new FilterOption(o.map(_.abs))

  def fSignum: FilterOption[Float] = new FilterOption(o.map(_.signum))

  def fMax(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Float, Float, Float](o, that.o, _ max _)
  def fMax(that: Float): FilterOption[Float] = new FilterOption(o.map(_ max that))

  def fMin(that: FilterOptionFloat): FilterOption[Float] = FilterOption[Float, Float, Float](o, that.o, _ min _)
  def fMin(that: Float): FilterOption[Float] = new FilterOption(o.map(_ min that))

  def fToDouble: FilterOption[Double] = new FilterOption(o.map(_.toDouble))

  def fLt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Float, Int, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Float, Long, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Float, Float, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Float, Double, Boolean](o, that.o, _ < _)
  def fLt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ < that))

  def fGt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Float, Int, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Float, Long, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Float, Float, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Float, Double, Boolean](o, that.o, _ > _)
  def fGt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ > that))

  def fLe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Float, Int, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Float, Long, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Float, Float, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Float, Double, Boolean](o, that.o, _ <= _)
  def fLe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))

  def fGe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Float, Int, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Float, Long, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Float, Float, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Float, Double, Boolean](o, that.o, _ >= _)
  def fGe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))

  // def fRound: FilterOption[Int] = new FilterOption[Int](o.map(scala.math.round))
}

class FilterOptionDouble(val o: Option[Double]) {
  def fPlus(that: FilterOptionInt): FilterOption[Double] = FilterOption[Double, Int, Double](o, that.o, _ + _)
  def fPlus(that: FilterOptionLong): FilterOption[Double] = FilterOption[Double, Long, Double](o, that.o, _ + _)
  def fPlus(that: FilterOptionFloat): FilterOption[Double] = FilterOption[Double, Float, Double](o, that.o, _ + _)
  def fPlus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, _ + _)
  def fPlus(that: Int): FilterOption[Double] = new FilterOption(o.map(_ + that))
  def fPlus(that: Long): FilterOption[Double] = new FilterOption(o.map(_ + that))
  def fPlus(that: Float): FilterOption[Double] = new FilterOption(o.map(_ + that))
  def fPlus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ + that))

  def fMinus(that: FilterOptionInt): FilterOption[Double] = FilterOption[Double, Int, Double](o, that.o, _ - _)
  def fMinus(that: FilterOptionLong): FilterOption[Double] = FilterOption[Double, Long, Double](o, that.o, _ - _)
  def fMinus(that: FilterOptionFloat): FilterOption[Double] = FilterOption[Double, Float, Double](o, that.o, _ - _)
  def fMinus(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, _ - _)
  def fMinus(that: Int): FilterOption[Double] = new FilterOption(o.map(_ - that))
  def fMinus(that: Long): FilterOption[Double] = new FilterOption(o.map(_ - that))
  def fMinus(that: Float): FilterOption[Double] = new FilterOption(o.map(_ - that))
  def fMinus(that: Double): FilterOption[Double] = new FilterOption(o.map(_ - that))

  def fTimes(that: FilterOptionInt): FilterOption[Double] = FilterOption[Double, Int, Double](o, that.o, _ * _)
  def fTimes(that: FilterOptionLong): FilterOption[Double] = FilterOption[Double, Long, Double](o, that.o, _ * _)
  def fTimes(that: FilterOptionFloat): FilterOption[Double] = FilterOption[Double, Float, Double](o, that.o, _ * _)
  def fTimes(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, _ * _)
  def fTimes(that: Int): FilterOption[Double] = new FilterOption(o.map(_ * that))
  def fTimes(that: Long): FilterOption[Double] = new FilterOption(o.map(_ * that))
  def fTimes(that: Float): FilterOption[Double] = new FilterOption(o.map(_ * that))
  def fTimes(that: Double): FilterOption[Double] = new FilterOption(o.map(_ * that))

  def fDiv(that: FilterOptionInt): FilterOption[Double] = FilterOption[Double, Int, Double](o, that.o, _ / _)
  def fDiv(that: FilterOptionLong): FilterOption[Double] = FilterOption[Double, Long, Double](o, that.o, _ / _)
  def fDiv(that: FilterOptionFloat): FilterOption[Double] = FilterOption[Double, Float, Double](o, that.o, _ / _)
  def fDiv(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, _ / _)
  def fDiv(that: Int): FilterOption[Double] = new FilterOption(o.map(_ / that))
  def fDiv(that: Long): FilterOption[Double] = new FilterOption(o.map(_ / that))
  def fDiv(that: Float): FilterOption[Double] = new FilterOption(o.map(_ / that))
  def fDiv(that: Double): FilterOption[Double] = new FilterOption(o.map(_ / that))

  def fMod(that: FilterOptionInt): FilterOption[Double] = FilterOption[Double, Int, Double](o, that.o, _ % _)
  def fMod(that: FilterOptionLong): FilterOption[Double] = FilterOption[Double, Long, Double](o, that.o, _ % _)
  def fMod(that: FilterOptionFloat): FilterOption[Double] = FilterOption[Double, Float, Double](o, that.o, _ % _)
  def fMod(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, _ % _)
  def fMod(that: Int): FilterOption[Double] = new FilterOption(o.map(_ % that))
  def fMod(that: Long): FilterOption[Double] = new FilterOption(o.map(_ % that))
  def fMod(that: Float): FilterOption[Double] = new FilterOption(o.map(_ % that))
  def fMod(that: Double): FilterOption[Double] = new FilterOption(o.map(_ % that))

  def unary_-(): FilterOption[Double] = new FilterOption(o.map(- _))
  def unary_+(): FilterOption[Double] = new FilterOption(o)

  def fAbs: FilterOption[Double] = new FilterOption(o.map(_.abs))

  def fSignum: FilterOption[Double] = new FilterOption(o.map(_.signum))

  def fMax(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, _ max _)
  def fMax(that: Double): FilterOption[Double] = new FilterOption(o.map(_ max that))

  def fMin(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, _ min _)
  def fMin(that: Double): FilterOption[Double] = new FilterOption(o.map(_ min that))

  def fLt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Double, Int, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Double, Long, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Double, Float, Boolean](o, that.o, _ < _)
  def fLt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Double, Double, Boolean](o, that.o, _ < _)
  def fLt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ < that))
  def fLt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ < that))

  def fGt(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Double, Int, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Double, Long, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Double, Float, Boolean](o, that.o, _ > _)
  def fGt(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Double, Double, Boolean](o, that.o, _ > _)
  def fGt(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ > that))
  def fGt(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ > that))

  def fLe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Double, Int, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Double, Long, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Double, Float, Boolean](o, that.o, _ <= _)
  def fLe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Double, Double, Boolean](o, that.o, _ <= _)
  def fLe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))
  def fLe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ <= that))

  def fGe(that: FilterOptionInt): FilterOption[Boolean] = FilterOption[Double, Int, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionLong): FilterOption[Boolean] = FilterOption[Double, Long, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionFloat): FilterOption[Boolean] = FilterOption[Double, Float, Boolean](o, that.o, _ >= _)
  def fGe(that: FilterOptionDouble): FilterOption[Boolean] = FilterOption[Double, Double, Boolean](o, that.o, _ >= _)
  def fGe(that: Int): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Long): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Float): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))
  def fGe(that: Double): FilterOption[Boolean] = new FilterOption(o.map(_ >= that))

  /*
  def fPow(that: FilterOptionDouble): FilterOption[Double] = FilterOption[Double, Double, Double](o, that.o, scala.math.pow)
  def fPow(that: Double): FilterOption[Double] = new FilterOption(o.map(scala.math.pow(_, that)))

  def fCbrt: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.cbrt))
  def fCeil: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.ceil))
  def fExp: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.exp))
  def fFloor: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.floor))
  def fLog: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.log))
  def fLog10: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.log10))
  def fRint: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.rint))
  def fRound: FilterOption[Long] = new FilterOption[Long](o.map(scala.math.round))
  def fSqrt: FilterOption[Double] = new FilterOption[Double](o.map(scala.math.sqrt))
  */
}

object FilterUtils {
  implicit def toFilterString(s: String): FilterString = new FilterString(s)

  type FilterGenotypeWithSA = ((IndexedSeq[AnnotationData], IndexedSeq[String]) =>
    ((Variant, AnnotationData) => ((Int, Genotype) => FilterOption[Boolean])))
  type FilterGenotypePostSA = (Variant, AnnotationData) => ((Int, Genotype) => FilterOption[Boolean])

  implicit def toAnnotationValueString(s: String): AnnotationValueString = new AnnotationValueString(s)

  implicit def toFilterOption[T](t: T): FilterOption[T] = new FilterOption(Some(t))

  implicit def toFilterOptionBoolean(b: Boolean): FilterOptionBoolean = new FilterOptionBoolean(Some(b))
  implicit def toFilterOptionBoolean(fo: FilterOption[Boolean]): FilterOptionBoolean = new FilterOptionBoolean(fo.o)

  implicit def toFilterOptionString(s: String): FilterOptionString = new FilterOptionString(Some(s))
  implicit def toFilterOptionString(fo: FilterOption[String]): FilterOptionString = new FilterOptionString(fo.o)

  implicit def toFilterOptionArray[T](a: Array[T]): FilterOptionArray[T] = new FilterOptionArray(Some(a))
  implicit def toFilterOptionArray[T](fo: FilterOption[Array[T]]): FilterOptionArray[T] = new FilterOptionArray[T](fo.o)

  implicit def toFilterOptionSet[T](s: Set[T]): FilterOptionSet[T] = new FilterOptionSet(Some(s))
  implicit def toFilterOptionSet[T](fo: FilterOption[Set[T]]): FilterOptionSet[T] = new FilterOptionSet[T](fo.o)

  implicit def toFilterOptionInt(i: Int): FilterOptionInt = new FilterOptionInt(Some(i))
  implicit def toFilterOptionInt(fo: FilterOption[Int]): FilterOptionInt = new FilterOptionInt(fo.o)

  implicit def toFilterOptionLong(l: Long): FilterOptionLong = new FilterOptionLong(Some(l))
  implicit def toFilterOptionLong(fo: FilterOption[Long]): FilterOptionLong = new FilterOptionLong(fo.o)

  implicit def toFilterOptionFloat(f: Float): FilterOptionFloat = new FilterOptionFloat(Some(f))
  implicit def toFilterOptionFloat(fo: FilterOption[Float]): FilterOptionFloat = new FilterOptionFloat(fo.o)

  implicit def toFilterOptionDouble(v: Double): FilterOptionDouble = new FilterOptionDouble(Some(v))
  implicit def toFilterOptionDouble(fo: FilterOption[Double]): FilterOptionDouble = new FilterOptionDouble(fo.o)
}

object Filter {
  def keepThis(fo: FilterOption[Boolean], keep: Boolean): Boolean =
    fo.o match {
      case Some(b) => if (keep) b else !b
      case None => false
    }
}

object FilterTransformer {
  val nameMap = Map(
    "toRealInt" -> "toInt", "toRealDouble" -> "toDouble",
    "$eq$eq" -> "fEq", "$bang$eq" -> "nNotEq", "apply" -> "fApply", "$amp$amp" -> "fAnd", "$bar$bar" -> "fOr",
    "$plus" -> "fPlus", "$minus" -> "fMinus", "$times" -> "fTimes", "$div" -> "fDiv", "$percent" -> "fMod",
    "abs" -> "fAbs", "signum" -> "fSignum", "max" -> "fMax", "min" -> "fMin", "contains" -> "fContains",
    "toInt" -> "fToInt", "toLong" -> "fToLong", "toFloat" -> "fToFloat", "toDouble" -> "fToDouble",
    "$less" -> "fLt", "$greater" -> "fGt", "$less$eq" -> "fLe", "$greater$eq" -> "fGe")
}

class FilterTransformer(m: Map[String, String]) extends Transformer {
  override def transform(t: Tree): Tree = t match {
    case Select(exp, TermName(n)) =>
      m.get(n) match {
        case Some(newName) => {println(s"match = ($n, $t)"); Select(transform(exp), TermName(newName))}
        case None => super.transform(t)
      }
    case _ => super.transform(t)
  }
}

class FilterVariantCondition(cond: String, vas: AnnotationSignatures)
  extends Evaluator[(Variant, AnnotationData) => FilterOption[Boolean]]({
    s"""(v: org.broadinstitute.hail.variant.Variant,
       |  __va: org.broadinstitute.hail.annotations.AnnotationData) => {
       |  import org.broadinstitute.hail.methods.FilterUtils._
       |  import org.broadinstitute.hail.methods.FilterOption
       |  ${signatures(vas, "__vaClass")}
       |  ${instantiate("va", "__vaClass", "__va")};
       |  {$cond}: FilterOption[Boolean]
       |}
    """.stripMargin
  }) {
  def apply(v: Variant, va: AnnotationData): FilterOption[Boolean] = eval()(v, va)
}

class FilterSampleCondition(cond: String, sas: AnnotationSignatures)
  extends Evaluator[(Sample, AnnotationData) => FilterOption[Boolean]](
    s"""(s: org.broadinstitute.hail.variant.Sample,
       |  __sa: org.broadinstitute.hail.annotations.AnnotationData) => {
       |  import org.broadinstitute.hail.methods.FilterUtils._
       |  import org.broadinstitute.hail.methods.FilterOption
       |  ${signatures(sas, "__saClass")}
       |  ${instantiate("sa", "__saClass", "__sa")};
       |  {$cond}: FilterOption[Boolean]
       |}
    """.stripMargin) {
  def apply(s: Sample, sa: AnnotationData): FilterOption[Boolean] = eval()(s, sa)
}

class FilterGenotypeCondition(cond: String, metadata: VariantMetadata)
  extends EvaluatorWithTransformation[FilterGenotypeWithSA, FilterGenotypePostSA](
    s"""(__sa: IndexedSeq[org.broadinstitute.hail.annotations.AnnotationData],
       |  __ids: IndexedSeq[String]) => {
       |  import org.broadinstitute.hail.methods.FilterUtils._
       |  import org.broadinstitute.hail.methods.FilterOption
       |  ${signatures(metadata.sampleAnnotationSignatures, "__saClass")}
       |  ${instantiateIndexedSeq("__saArray", "__saClass", "__sa")}
       |  (v: org.broadinstitute.hail.variant.Variant,
       |    __va: org.broadinstitute.hail.annotations.AnnotationData) => {
       |    ${signatures(metadata.variantAnnotationSignatures, "__vaClass")}
       |    ${instantiate("va", "__vaClass",  "__va")}
       |    (__sIndex: Int,
       |     g: org.broadinstitute.hail.variant.Genotype) => {
       |      val sa = __saArray(__sIndex)
       |      val s = org.broadinstitute.hail.variant.Sample(__ids(__sIndex));
       |      {$cond}: FilterOption[Boolean]
       |    }
       |  }
       |}
      """.stripMargin,
    t => t(metadata.sampleAnnotations, metadata.sampleIds)) {
  def apply(v: Variant, va: AnnotationData)(sIndex: Int, g: Genotype): FilterOption[Boolean] =
    eval()(v, va)(sIndex, g)
}
