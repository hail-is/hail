package is.hail.utils

import is.hail.annotations._
import is.hail.check._
import is.hail.expr.types.{TBoolean, TInterval, TStruct}
import org.json4s.JValue
import org.json4s.JsonAST.JObject

import scala.collection.mutable
import scala.language.implicitConversions
import scala.reflect.ClassTag

case class Interval(start: Any, end: Any, includesStart: Boolean, includesEnd: Boolean) extends Serializable {

  def contains(pord: ExtendedOrdering, position: Any): Boolean = {
    val compareStart = pord.compare(position, start)
    if (compareStart > 0 || (compareStart == 0 && includesStart)) {
      val compareEnd = pord.compare(position, end)
      compareEnd < 0 || (compareEnd == 0 && includesEnd)
    } else
      false
  }

  def includes(pord: ExtendedOrdering, other: Interval): Boolean =
    other.definitelyEmpty(pord) || ({
      val cstart = pord.compare(this.start, other.start)
      cstart < 0 || (cstart == 0 && (!other.includesStart || this.includesStart))
    } && {
      val cend = pord.compare(this.end, other.end)
      cend > 0 || (cend == 0 && (!other.includesEnd || this.includesEnd))
    })

  def mayOverlap(pord: ExtendedOrdering, other: Interval): Boolean = {
    !definitelyDisjoint(pord, other)
  }

  def isAbovePosition(pord: ExtendedOrdering, p: Any): Boolean =
    definitelyEmpty(pord) || {
      val c = pord.compare(p, start)
      c < 0 || (c == 0 && !includesStart)
    }

  def isBelowPosition(pord: ExtendedOrdering, p: Any): Boolean =
    definitelyEmpty(pord) || {
      val c = pord.compare(p, end)
      c > 0 || (c == 0 && !includesEnd)
    }

  def definitelyDisjoint(pord: ExtendedOrdering, other: Interval): Boolean =
    definitelyEmpty(pord) || other.definitelyEmpty(pord) ||
      isBelow(pord, other) || isAbove(pord, other)

  private var _emptyDefined: Boolean = false
  private var _empty: Boolean = false

  def definitelyEmpty(pord: ExtendedOrdering): Boolean = {
    if (!_emptyDefined) {
      _empty = if (includesStart && includesEnd) pord.gt(start, end) else pord.gteq(start, end)
      _emptyDefined = true
    }
    _empty
  }

  def copy(start: Any = start, end: Any = end, includesStart: Boolean = includesStart, includesEnd: Boolean = includesEnd): Interval =
    Interval(start, end, includesStart, includesEnd)

  def toJSON(f: (Any) => JValue): JValue =
    JObject("start" -> f(start),
      "end" -> f(end),
      "includeStart" -> TBoolean().toJSON(includesStart),
      "includeEnd" -> TBoolean().toJSON(includesEnd))

  def isBelow(pord: ExtendedOrdering, other: Interval): Boolean =
    this.definitelyEmpty(pord) || other.definitelyEmpty(pord) || {
      val c = pord.compare(this.end, other.start)
      c < 0 || (c == 0 && (!this.includesEnd || !other.includesStart))
    }

  def isAbove(pord: ExtendedOrdering, other: Interval): Boolean =
    this.definitelyEmpty(pord) || other.definitelyEmpty(pord) || {
      val c = pord.compare(this.start, other.end)
      c > 0 || (c == 0 && (!this.includesStart || !other.includesEnd))
    }

  def canMergeWith(pord: ExtendedOrdering, other: Interval): Boolean =
    this.definitelyEmpty(pord) || other.definitelyEmpty(pord) || ({
      val c = pord.compare(this.start, other.end)
      c < 0 || (c == 0 && (this.includesStart || other.includesEnd))
    } && {
      val c = pord.compare(this.end, other.start)
      c > 0 || (c == 0 && (this.includesEnd || other.includesStart))
    })

  def merge(pord: ExtendedOrdering, other: Interval): Option[Interval] = {
    if (canMergeWith(pord, other)) {
      if (other.definitelyEmpty(pord))
        Some(this)
      else if (this.definitelyEmpty(pord))
        Some(other)
      else {
        val min = Interval.ordering(pord, startPrimary = true).min(this, other).asInstanceOf[Interval]
        val max = Interval.ordering(pord, startPrimary = false).max(this, other).asInstanceOf[Interval]
        Some(Interval(min.start, max.end, min.includesStart, max.includesEnd))
      }
    } else
      None
  }

  def intersect(pord: ExtendedOrdering, other: Interval): Interval = {
    if (mayOverlap(pord, other)) {
      val s = Interval.ordering(pord, startPrimary = true).max(this, other).asInstanceOf[Interval]
      val e = Interval.ordering(pord, startPrimary = false).min(this, other).asInstanceOf[Interval]
      Interval(s.start, e.end, s.includesStart, e.includesEnd)
    } else
      Interval(start, start, false, false)
  }

  override def toString: String = (if (includesStart) "[" else "(") + start + "-" + end + (if (includesEnd) "]" else ")")
}

object Interval {
  def gen[P](pord: ExtendedOrdering, pgen: Gen[P]): Gen[Interval] =
    Gen.zip(pgen, pgen, Gen.coin(), Gen.coin())
      .map { case (x, y, s, e) =>
        if (pord.lt(x, y))
          Interval(x, y, s, e)
        else
          Interval(y, x, s, e)
      }

  def ordering(pord: ExtendedOrdering, startPrimary: Boolean): ExtendedOrdering = new ExtendedOrdering {
    private def compareStart(x: Interval, y: Interval, missingGreatest: Boolean, next: (Interval, Interval) => Int): Int = {
      val c = pord.compare(x.start, y.start, missingGreatest)
      if (c != 0)
        c
      else if (x.includesStart != y.includesStart)
        if (x.includesStart) -1 else 1
      else next(x, y)
    }

    private def compareEnd(x: Interval, y: Interval, missingGreatest: Boolean, next: (Interval, Interval) => Int): Int = {
      val c = pord.compare(x.end, y.end, missingGreatest)
      if (c != 0)
        c
      else if (x.includesEnd != y.includesEnd)
        if (x.includesEnd) 1 else -1
      else next(x, y)
    }

    def compareNonnull(x: Any, y: Any, missingGreatest: Boolean): Int = {
      val xi = x.asInstanceOf[Interval]
      val yi = y.asInstanceOf[Interval]

      if (startPrimary)
        compareStart(xi, yi, missingGreatest, compareEnd(_, _, missingGreatest, (_, _) => 0))
      else
        compareEnd(xi, yi, missingGreatest, compareStart(_, _, missingGreatest, (_, _) => 0))
    }
  }

  def fromRegionValue(iType: TInterval, region: Region, offset: Long): Interval = {
    val ur = new UnsafeRow(iType.fundamentalType.asInstanceOf[TStruct], region, offset)
    Interval(ur.get(0), ur.get(1), ur.getAs[Boolean](2), ur.getAs[Boolean](3))
  }
}

case class IntervalTree[U: ClassTag](root: Option[IntervalTreeNode[U]]) extends
  Traversable[(Interval, U)] with Serializable {
  override def size: Int = root.map(_.size).getOrElse(0)

  def definitelyEmpty(pord: ExtendedOrdering): Boolean = root.forall(_.definitelyEmpty(pord))

  def contains(pord: ExtendedOrdering, position: Any): Boolean = root.exists(_.contains(pord, position))

  def probablyOverlaps(pord: ExtendedOrdering, interval: Interval): Boolean = root.exists(_.probablyOverlaps(pord, interval))

  def definitelyDisjoint(pord: ExtendedOrdering, interval: Interval): Boolean = root.forall(_.definitelyDisjoint(pord, interval))

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
      while (unpruned(i).definitelyEmpty(pord)) {
        i += 1
        if (i == unpruned.length) {
          return new IntervalTree[Unit](None)
        }
      }
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

  def definitelyEmpty(pord: ExtendedOrdering): Boolean = {
    left.forall(_.definitelyEmpty(pord)) &&
      i.definitelyEmpty(pord) &&
      right.forall(_.definitelyEmpty(pord))
  }

  def contains(pord: ExtendedOrdering, position: Any): Boolean = {
    range.contains(pord, position) &&
      (left.exists(_.contains(pord, position)) ||
        (pord.gteq(position, i.start) &&
          (i.contains(pord, position) ||
            right.exists(_.contains(pord, position)))))
  }

  def probablyOverlaps(pord: ExtendedOrdering, interval: Interval): Boolean = {
    !definitelyDisjoint(pord, interval)
  }

  def definitelyDisjoint(pord: ExtendedOrdering, interval: Interval): Boolean =
    range.definitelyDisjoint(pord, interval) ||
      (left.forall(_.definitelyDisjoint(pord, interval)) &&
        i.definitelyDisjoint(pord, interval) &&
        right.forall(_.definitelyDisjoint(pord, interval)))

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
    if (range.mayOverlap(pord, interval)) {
      left.foreach(_.queryOverlappingValues(pord, b, interval))
      if (i.mayOverlap(pord, interval))
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
