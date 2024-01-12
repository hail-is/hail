package is.hail.utils

import is.hail.annotations._
import is.hail.check._
import is.hail.types.virtual.TBoolean

import org.json4s.JsonAST.JObject
import org.json4s.JValue

import scala.language.implicitConversions

import org.apache.spark.sql.Row

case class IntervalEndpoint(point: Any, sign: Int) extends Serializable {
  require(sign == -1 || sign == 1)

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

/** 'Interval' has an implicit precondition that 'start' and 'end' either have the same type, or are
  * of compatible 'TBaseStruct' types, i.e. their types agree on all fields up to the min of their
  * lengths. Moreover, it assumes that the interval is well formed, as coded in 'Interval.isValid',
  * roughly meaning that start is less than end. Each method assumes that the 'pord' parameter is
  * compatible with the endpoints, and with 'p' or the endpoints of 'other'.
  *
  * Precisely, 'Interval' assumes that there exists a Hail type 't: Type' such that either
  *   - 't: TBaseStruct', and 't.relaxedTypeCheck(left)', 't.relaxedTypeCheck(right), and
  *     't.ordering.intervalEndpointOrdering.lt(left, right)', or
  *   - 't.typeCheck(left)', 't.typeCheck(right)', and 't.ordering.lt(left, right)'
  *
  * Moreover, every method on 'Interval' taking a 'pord' has the precondition that there exists a
  * Hail type 't: Type' such that 'pord' was constructed by 't.ordering', and either
  *   - 't: TBaseStruct' and 't.relaxedTypeCheck(x)', or
  *   - 't.typeCheck(x)', where 'x' is each of 'left', 'right', 'p', 'other.left', and 'other.right'
  *     as appropriate. In the case 't: TBaseStruct', 't' could be replaced by any 't2' such that
  *     't.isPrefixOf(t2)' without changing the behavior.
  */
class Interval(val left: IntervalEndpoint, val right: IntervalEndpoint) extends Serializable {
  require(left != null)
  require(right != null)
  def start: Any = left.point

  def end: Any = right.point

  def includesStart: Boolean = left.sign < 0

  def includesEnd: Boolean = right.sign > 0

  private def ext(pord: ExtendedOrdering): IntervalEndpointOrdering = pord.intervalEndpointOrdering

  def contains(pord: ExtendedOrdering, p: Any): Boolean =
    ext(pord).compare(left, p) < 0 && ext(pord).compare(right, p) > 0

  def includes(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).compare(this.left, other.left) <= 0 && ext(pord).compare(this.right, other.right) >= 0

  def overlaps(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).compare(this.left, other.right) < 0 && ext(pord).compare(this.right, other.left) > 0

  def isAbovePosition(pord: ExtendedOrdering, p: Any): Boolean =
    ext(pord).compare(left, p) > 0

  def isBelowPosition(pord: ExtendedOrdering, p: Any): Boolean =
    ext(pord).compare(right, p) < 0

  def isDisjointFrom(pord: ExtendedOrdering, other: Interval): Boolean =
    !overlaps(pord, other)

  def copy(
    start: Any = start,
    end: Any = end,
    includesStart: Boolean = includesStart,
    includesEnd: Boolean = includesEnd,
  ): Interval =
    Interval(start, end, includesStart, includesEnd)

  def extendLeft(newLeft: IntervalEndpoint): Interval = Interval(newLeft, right)

  def extendRight(newRight: IntervalEndpoint): Interval = Interval(left, newRight)

  def toJSON(f: (Any) => JValue): JValue =
    JObject(
      "start" -> f(start),
      "end" -> f(end),
      "includeStart" -> TBoolean.toJSON(includesStart),
      "includeEnd" -> TBoolean.toJSON(includesEnd),
    )

  def isBelow(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).compare(this.right, other.left) <= 0

  def isAbove(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).compare(this.left, other.right) >= 0

  def abutts(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).compare(this.left, other.right) == 0 || ext(pord).compare(this.right, other.left) == 0

  def canMergeWith(pord: ExtendedOrdering, other: Interval): Boolean =
    ext(pord).compare(this.left, other.right) <= 0 && ext(pord).compare(this.right, other.left) >= 0

  def merge(pord: ExtendedOrdering, other: Interval): Option[Interval] =
    if (canMergeWith(pord, other))
      Some(hull(pord, other))
    else
      None

  def hull(pord: ExtendedOrdering, other: Interval): Interval =
    Interval(
      // min(this.left, other.left)
      if (ext(pord).compare(this.left, other.left) < 0)
        this.left
      else
        other.left,
      //  max(this.right, other.right)
      if (ext(pord).compare(this.right, other.right) < 0)
        other.right
      else
        this.right,
    )

  def intersect(pord: ExtendedOrdering, other: Interval): Option[Interval] =
    if (overlaps(pord, other)) {
      Some(Interval(
        // max(this.left, other.left)
        if (ext(pord).compare(this.left, other.left) < 0)
          other.left
        else
          this.left,
        // min(this.right, other.right)
        if (ext(pord).compare(this.right, other.right) < 0)
          this.right
        else
          other.right,
      ))
    } else
      None

  def coarsen(newKeyLen: Int): Interval =
    Interval(left.coarsenLeft(newKeyLen), right.coarsenRight(newKeyLen))

  override def toString: String =
    (if (includesStart) "[" else "(") + start + "-" + end + (if (includesEnd) "]" else ")")

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

  def orNone(
    pord: ExtendedOrdering,
    start: Any,
    end: Any,
    includesStart: Boolean,
    includesEnd: Boolean,
  ): Option[Interval] =
    if (isValid(pord, start, end, includesStart, includesEnd))
      Some(Interval(start, end, includesStart, includesEnd))
    else
      None

  def orNone(pord: ExtendedOrdering, left: IntervalEndpoint, right: IntervalEndpoint)
    : Option[Interval] =
    orNone(pord, left.point, right.point, left.sign < 0, right.sign > 0)

  def isValid(
    pord: ExtendedOrdering,
    start: Any,
    end: Any,
    includesStart: Boolean,
    includesEnd: Boolean,
  ): Boolean = {
    val (left, right) = toIntervalEndpoints(start, end, includesStart, includesEnd)
    pord.intervalEndpointOrdering.compare(left, right) < 0
  }

  def toIntervalEndpoints(
    start: Any,
    end: Any,
    includesStart: Boolean,
    includesEnd: Boolean,
  ): (IntervalEndpoint, IntervalEndpoint) =
    (
      IntervalEndpoint(start, if (includesStart) -1 else 1),
      IntervalEndpoint(end, if (includesEnd) 1 else -1),
    )

  def gen[P](pord: ExtendedOrdering, pgen: Gen[P]): Gen[Interval] =
    Gen.zip(pgen, pgen, Gen.coin(), Gen.coin())
      .filter { case (x, y, s, e) => pord.compare(x, y) != 0 || (s && e) }
      .map { case (x, y, s, e) =>
        if (pord.compare(x, y) < 0)
          Interval(x, y, s, e)
        else
          Interval(y, x, s, e)
      }

  def ordering(pord: ExtendedOrdering, startPrimary: Boolean, _missingEqual: Boolean = true)
    : ExtendedOrdering = new ExtendedOrdering {
    val missingEqual = _missingEqual

    override def compareNonnull(x: Any, y: Any): Int = {
      val xi = x.asInstanceOf[Interval]
      val yi = y.asInstanceOf[Interval]

      if (startPrimary) {
        val c = pord.intervalEndpointOrdering.compareNonnull(xi.left, yi.left)
        if (c != 0) c else pord.intervalEndpointOrdering.compareNonnull(xi.right, yi.right)
      } else {
        val c = pord.intervalEndpointOrdering.compareNonnull(xi.right, yi.right)
        if (c != 0) c else pord.intervalEndpointOrdering.compareNonnull(xi.left, yi.left)
      }
    }
  }

  def union(xs: IndexedSeq[Interval], ord: IntervalEndpointOrdering): Array[Interval] = {

    val sorted = xs.sortBy(_.left: Any)(ord.toOrdering)

    val ab = new BoxedArrayBuilder[Interval]()
    var i = 0
    while (i < sorted.length) {
      var interval = sorted(i)
      i += 1
      while (i < sorted.length && ord.gteq(interval.right, sorted(i).left)) {
        interval = Interval(interval.left, ordMax(interval.right, sorted(i).right, ord))
        i += 1
      }

      ab += interval
    }
    ab.result()
  }

  // assumes that both `x1` and `x2` are both sorted, non-overlapping interval sequences.
  def intersection(
    x1: IndexedSeq[Interval],
    x2: IndexedSeq[Interval],
    ord: IntervalEndpointOrdering,
  ): Array[Interval] = {

    var i = 0
    var j = 0
    val ab = new BoxedArrayBuilder[Interval]()

    while (!(i >= x1.length || j >= x2.length)) {
      val l = x1(i)
      val r = x2(j)

      if (ord.gteq(l.left, r.right))
        j += 1
      else if (ord.gteq(r.left, l.right))
        i += 1
      else {
        val overlap = Interval(ordMax(l.left, r.left, ord), ordMin(l.right, r.right, ord))
        ab += overlap
        if (ord.lt(l.right, r.right))
          i += 1
        else
          j += 1
      }
    }
    ab.result()
  }
}
