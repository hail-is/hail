package is.hail.utils

import is.hail.annotations._
import is.hail.check._
import is.hail.expr.types.{TBoolean, TInterval, TStruct}
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

/**
  * 'Interval' has an implicit precondition that 'start' and 'end' either have
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

case class IntervalTree[U: ClassTag](root: Option[IntervalTreeNode[U]]) extends
  Traversable[(Interval, U)] with Serializable {
  override def size: Int = root.map(_.size).getOrElse(0)

  def isEmpty(pord: ExtendedOrdering): Boolean = root.isEmpty

  def contains(pord: ExtendedOrdering, position: Any): Boolean = root.exists(_.contains(pord, position))

  def overlaps(pord: ExtendedOrdering, interval: Interval): Boolean = root.exists(_.overlaps(pord, interval))

  def isDisjointFrom(pord: ExtendedOrdering, interval: Interval): Boolean = root.forall(_.isDisjointFrom(pord, interval))

  def queryIntervals(pord: ExtendedOrdering, position: Any): Array[Interval] = {
    val b = Array.newBuilder[Interval]
    root.foreach(_.query(pord, b, position))
    b.result()
  }

  def queryValues(pord: ExtendedOrdering, position: Any): Array[U] = {
    val b = Array.newBuilder[U]
    root.foreach(_.queryValues(pord, b, position))
    b.result()
  }

  def queryOverlappingValues(pord: ExtendedOrdering, interval: Interval): Array[U] = {
    val b = Array.newBuilder[U]
    root.foreach(_.queryOverlappingValues(pord, b, interval))
    b.result()
  }

  def foreach[V](f: ((Interval, U)) => V) {
    root.foreach(_.foreach(f))
  }
}

object IntervalTree {
  def annotationTree[U: ClassTag](pord: ExtendedOrdering, values: Array[(Interval, U)]): IntervalTree[U] = {
    val iord = Interval.ordering(pord, startPrimary = true)
    val sorted = values.sortBy(_._1)(iord.toOrdering.asInstanceOf[Ordering[Interval]])
    new IntervalTree[U](fromSorted(pord, sorted, 0, sorted.length))
  }

  def apply(pord: ExtendedOrdering, intervals: Array[Interval]): IntervalTree[Unit] = {
    val iord = Interval.ordering(pord, startPrimary = true)
    val sorted = if (intervals.nonEmpty) {
      val unpruned = intervals.sorted(iord.toOrdering.asInstanceOf[Ordering[Interval]])
      var i = 0
      val ab = new ArrayBuilder[Interval](intervals.length)
      var tmp = unpruned(i)
      while (i < unpruned.length) {
        tmp.merge(pord, unpruned(i)) match {
          case Some(interval) =>
            tmp = interval
          case None =>
            ab += tmp
            tmp = unpruned(i)
        }
        i += 1
      }
      ab += tmp
      ab.result()
    } else intervals

    new IntervalTree[Unit](fromSorted(pord, sorted.map(i => (i, ())), 0, sorted.length))
  }

  def fromSorted[U: ClassTag](pord: ExtendedOrdering, intervals: Array[(Interval, U)]): IntervalTree[U] =
    new IntervalTree[U](fromSorted(pord, intervals, 0, intervals.length))

  private def fromSorted[U](pord: ExtendedOrdering, intervals: Array[(Interval, U)], start: Int, end: Int): Option[IntervalTreeNode[U]] = {
    if (start >= end)
      None
    else {
      val mid = (start + end) / 2
      val (i, v) = intervals(mid)
      val left = fromSorted(pord, intervals, start, mid)
      val right = fromSorted(pord, intervals, mid + 1, end)

      val min = left.map { inode => inode.range }.getOrElse(i)
      val eord = Interval.ordering(pord, startPrimary = false)
      val max = right.foldLeft(
        left.foldLeft(i) { (i2, n) =>
          eord.max(i2, n.range).asInstanceOf[Interval]
        }) { (i2, n) =>
        eord.max(i2, n.range).asInstanceOf[Interval]
      }

      Some(IntervalTreeNode(i, left, right,
        Interval(min.start, max.end, min.includesStart, max.includesEnd), v))
    }
  }

  def gen[T](pord: ExtendedOrdering, pgen: Gen[T]): Gen[IntervalTree[Unit]] = {
    Gen.buildableOf[Array](Interval.gen(pord, pgen)).map(a => IntervalTree.apply(pord, a))
  }
}

case class IntervalTreeNode[U](i: Interval,
  left: Option[IntervalTreeNode[U]],
  right: Option[IntervalTreeNode[U]],
  range: Interval, value: U) extends Traversable[(Interval, U)] {

  override val size: Int =
    left.map(_.size).getOrElse(0) + right.map(_.size).getOrElse(0) + 1

  def contains(pord: ExtendedOrdering, position: Any): Boolean = {
    range.contains(pord, position) &&
      (left.exists(_.contains(pord, position)) ||
        (pord.gteq(position, i.start) &&
          (i.contains(pord, position) ||
            right.exists(_.contains(pord, position)))))
  }

  def overlaps(pord: ExtendedOrdering, interval: Interval): Boolean = {
    !isDisjointFrom(pord, interval)
  }

  def isDisjointFrom(pord: ExtendedOrdering, interval: Interval): Boolean =
    range.isDisjointFrom(pord, interval) ||
      (left.forall(_.isDisjointFrom(pord, interval)) &&
        i.isDisjointFrom(pord, interval) &&
        right.forall(_.isDisjointFrom(pord, interval)))

  def query(pord: ExtendedOrdering, b: mutable.Builder[Interval, _], position: Any) {
    if (range.contains(pord, position)) {
      left.foreach(_.query(pord, b, position))
      if (i.contains(pord, position))
        b += i
      right.foreach(_.query(pord, b, position))
    }
  }

  def queryValues(pord: ExtendedOrdering, b: mutable.Builder[U, _], position: Any) {
    if (range.contains(pord, position)) {
      left.foreach(_.queryValues(pord, b, position))
      if (i.contains(pord, position))
        b += value
      right.foreach(_.queryValues(pord, b, position))
    }
  }

  def queryOverlappingValues(pord: ExtendedOrdering, b: mutable.Builder[U, _], interval: Interval) {
    if (range.overlaps(pord, interval)) {
      left.foreach(_.queryOverlappingValues(pord, b, interval))
      if (i.overlaps(pord, interval))
        b += value
      right.foreach(_.queryOverlappingValues(pord, b, interval))
    }
  }

  def foreach[V](f: ((Interval, U)) => V) {
    left.foreach(_.foreach(f))
    f((i, value))
    right.foreach(_.foreach(f))
  }
}
