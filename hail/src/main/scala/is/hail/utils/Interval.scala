package is.hail.utils

import is.hail.annotations._
import is.hail.check._
import is.hail.expr.types.virtual.{TBoolean, TStruct}
import org.apache.spark.sql.Row
import org.json4s.JValue
import org.json4s.JsonAST.JObject

import scala.collection.mutable
import scala.language.implicitConversions
import scala.reflect.ClassTag

case class IntervalEndpoint(point: Any, sign: Int) extends Serializable {
  require(-1 <= sign && sign <= 1)

  def coarsenLeft(newKeyLen: Int): IntervalEndpoint =
    coarsen(newKeyLen, -1)

  def coarsenRight(newKeyLen: Int): IntervalEndpoint =
    coarsen(newKeyLen, 1)

  private def coarsen(newKeyLen: Int, sign: Int): IntervalEndpoint = {
    val row = point.asInstanceOf[Row]
    if (row.size > newKeyLen)
      IntervalEndpoint(row.truncate(newKeyLen), sign)
    else
      this
  }
}

/** 'Interval' has an implicit precondition that 'start' and 'end' either have
  * the same type, or are of compatible 'TBaseStruct' types, i.e. their types
  * agree on all fields up to the min of their lengths. Moreover, it assumes
  * that the interval is well formed, as coded in 'Interval.isValid', roughly
  * meaning that start is less than end. Each method assumes that the 'pord'
  * parameter is compatible with the endpoints, and with 'p' or the endpoints
  * of 'other'.
  *
  * Precisely, 'Interval' assumes that there exists a Hail type 't: Type' such
  * that either
  * - 't: TBaseStruct', and 't.relaxedTypeCheck(left)', 't.relaxedTypeCheck(right),
  *   and 't.ordering.intervalEndpointOrdering.lt(left, right)', or
  * - 't.typeCheck(left)', 't.typeCheck(right)', and 't.ordering.lt(left, right)'
  *
  * Moreover, every method on 'Interval' taking a 'pord' has the precondition
  * that there exists a Hail type 't: Type' such that 'pord' was constructed by
  * 't.ordering', and either
  * - 't: TBaseStruct' and 't.relaxedTypeCheck(x)', or
  * - 't.typeCheck(x)',
  * where 'x' is each of 'left', 'right', 'p', 'other.left', and 'other.right'
  * as appropriate. In the case 't: TBaseStruct', 't' could be replaced by any
  * 't2' such that 't.isPrefixOf(t2)' without changing the behavior.
  */
class Interval(val left: IntervalEndpoint, val right: IntervalEndpoint) extends Serializable {
  def start: Any = left.point
  def end: Any = right.point
  def includesStart = left.sign < 0
  def includesEnd = right.sign > 0

  private def ext(pord: ExtendedOrdering): ExtendedOrdering = pord.intervalEndpointOrdering

  def contains(pord: ExtendedOrdering, p: Any): Boolean =
    ext(pord).lt(left, p) && ext(pord).gt(right, p)

  def includes(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).lteq(this.left, other.left) && ext(pord).gteq(this.right, other.right)

  def overlaps(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).lt(this.left, other.right) && ext(pord).gt(this.right, other.left)

  def isAbovePosition(pord: ExtendedOrdering, p: Any): Boolean =
    ext(pord).gt(left, p)

  def isBelowPosition(pord: ExtendedOrdering, p: Any): Boolean =
    ext(pord).lt(right, p)

  def isDisjointFrom(pord: ExtendedOrdering, other: Interval): Boolean =
    !overlaps(pord, other)

  def copy(start: Any = start, end: Any = end, includesStart: Boolean = includesStart, includesEnd: Boolean = includesEnd): Interval =
    Interval(start, end, includesStart, includesEnd)

  def extendLeft(newLeft: IntervalEndpoint): Interval = Interval(newLeft, right)
  def extendRight(newRight: IntervalEndpoint): Interval = Interval(left, newRight)

  def toJSON(f: (Any) => JValue): JValue =
    JObject("start" -> f(start),
      "end" -> f(end),
      "includeStart" -> TBoolean().toJSON(includesStart),
      "includeEnd" -> TBoolean().toJSON(includesEnd))

  def isBelow(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).lteq(this.right, other.left)

  def isAbove(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).gteq(this.left, other.right)

  def abutts(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).equiv(this.left, other.right) || ext(pord).equiv(this.right, other.left)

  def canMergeWith(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).lteq(this.left, other.right) && ext(pord).gteq(this.right, other.left)

  def merge(pord: ExtendedOrdering, other: Interval): Option[Interval] =
    if (canMergeWith(pord, other))
      Some(hull(pord, other))
    else
      None

  def hull(pord: ExtendedOrdering, other: Interval): Interval =
    Interval(
      ext(pord).min(this.left, other.left).asInstanceOf[IntervalEndpoint],
      ext(pord).max(this.right, other.right).asInstanceOf[IntervalEndpoint])

  def intersect(pord: ExtendedOrdering, other: Interval): Option[Interval] =
    if (overlaps(pord, other)) {
      Some(Interval(
        ext(pord).max(this.left, other.left).asInstanceOf[IntervalEndpoint],
        ext(pord).min(this.right, other.right).asInstanceOf[IntervalEndpoint]))
    } else
      None

  def coarsen(newKeyLen: Int): Interval =
    Interval(left.coarsenLeft(newKeyLen), right.coarsenRight(newKeyLen))

  override def toString: String = (if (includesStart) "[" else "(") + start + "-" + end + (if (includesEnd) "]" else ")")

  override def equals(other: Any): Boolean = other match {
    case that: Interval => left == that.left && right == that.right
    case _ => false
  }

  override def hashCode(): Int = (left, right).##
}

object Interval {
  def apply(left: IntervalEndpoint, right: IntervalEndpoint): Interval =
    new Interval(left, right)

  def apply(start: Any, end: Any, includesStart: Boolean, includesEnd: Boolean): Interval = {
    val (left, right) = toIntervalEndpoints(start, end, includesStart, includesEnd)
    Interval(left, right)
  }

  def unapply(interval: Interval): Option[(Any, Any, Boolean, Boolean)] =
    Some((interval.start, interval.end, interval.includesStart, interval.includesEnd))

  def orNone(pord: ExtendedOrdering,
    start: Any, end: Any,
    includesStart: Boolean, includesEnd: Boolean
  ): Option[Interval] =
    if (isValid(pord, start, end, includesStart, includesEnd))
      Some(Interval(start, end, includesStart, includesEnd))
    else
      None

  def orNone(pord: ExtendedOrdering, left: IntervalEndpoint, right: IntervalEndpoint): Option[Interval] =
    orNone(pord, left.point, right.point, left.sign < 0, right.sign > 0)

  def isValid(pord: ExtendedOrdering,
    start: Any, end: Any,
    includesStart: Boolean, includesEnd: Boolean
  ): Boolean = {
    val (left, right) = toIntervalEndpoints(start, end, includesStart, includesEnd)
    pord.intervalEndpointOrdering.lt(left, right)
  }

  def toIntervalEndpoints(
    start: Any, end: Any,
    includesStart: Boolean, includesEnd: Boolean
  ): (IntervalEndpoint, IntervalEndpoint) =
    (IntervalEndpoint(start, if (includesStart) -1 else 1),
      IntervalEndpoint(end, if (includesEnd) 1 else -1))

  def gen[P](pord: ExtendedOrdering, pgen: Gen[P]): Gen[Interval] =
    Gen.zip(pgen, pgen, Gen.coin(), Gen.coin())
      .filter { case (x, y, s, e) => !pord.equiv(x, y) || (s && e) }
      .map { case (x, y, s, e) =>
        if (pord.lt(x, y))
          Interval(x, y, s, e)
        else
          Interval(y, x, s, e)
      }

  def ordering(pord: ExtendedOrdering, startPrimary: Boolean): ExtendedOrdering = new ExtendedOrdering {
    override def compareNonnull(x: Any, y: Any, missingGreatest: Boolean): Int = {
      val xi = x.asInstanceOf[Interval]
      val yi = y.asInstanceOf[Interval]

      if (startPrimary) {
        val c = pord.intervalEndpointOrdering.compare(xi.left, yi.left, missingGreatest)
        if (c != 0) c else pord.intervalEndpointOrdering.compare(xi.right, yi.right, missingGreatest)
      } else {
        val c = pord.intervalEndpointOrdering.compare(xi.right, yi.right, missingGreatest)
        if (c != 0) c else pord.intervalEndpointOrdering.compare(xi.left, yi.left, missingGreatest)
      }
    }
  }
}
