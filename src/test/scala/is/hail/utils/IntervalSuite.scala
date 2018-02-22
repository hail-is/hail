package is.hail.utils

import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.TInt32
import org.testng.annotations.Test

class IntervalSuite {

  val pord: ExtendedOrdering = TInt32().ordering

  // set of intervals chosen from 5 endpoints spans the space of relations
  // that two non-empty intervals can have with each other.
  val points: IndexedSeq[Int] = 1 to 5

  val test_intervals: IndexedSeq[SetInterval] =
    for {
      s <- points
      e <- points
      is <- Array(true, false)
      ie <- Array(true, false)
    } yield SetInterval(s, e, is, ie)

  val test_itrees: IndexedSeq[SetIntervalTree] =
    for {
      i1 <- test_intervals
      i2 <- test_intervals
    } yield SetIntervalTree(Array(i1, i2).zipWithIndex)

  def shouldOrShouldnt(bool: Boolean): String = if (bool) "should" else "shouldn't"

  @Test def intervalEmpty() {
    for (set_interval <- test_intervals) {
      val interval = set_interval.interval
      assert(interval.definitelyEmpty(pord) iff set_interval.definitelyEmpty(),
        s"$interval ${ shouldOrShouldnt(set_interval.definitelyEmpty()) } be empty")
    }
  }

  @Test def intervalContains() {
    for (set_interval <- test_intervals; p <- points) {
      val interval = set_interval.interval
      assert(interval.contains(pord, p) iff set_interval.contains(p),
        s"$p ${ shouldOrShouldnt(set_interval.contains(p)) } be in $interval")
    }
  }

  @Test def intervalOverlaps() {
    for (set_interval1 <- test_intervals; set_interval2 <- test_intervals) {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assert(set_interval1.probablyOverlaps(set_interval2) iff interval1.probablyOverlaps(pord, interval2),
        s"$interval1 ${ shouldOrShouldnt(set_interval1.probablyOverlaps(set_interval2)) } overlap with $interval2")
    }
  }

  @Test def intervalDisjoint() {
    for (set_interval1 <- test_intervals; set_interval2 <- test_intervals) {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assert(interval1.definitelyDisjoint(pord, interval2) iff set_interval1.definitelyDisjoint(set_interval2),
        s"$interval1 ${ shouldOrShouldnt(set_interval1.definitelyDisjoint(set_interval2)) } be disjoint with $interval2")
    }
  }

  @Test def intervalTreeContains() {
    for {
      set_itree <- test_itrees
      point <- points
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      assert(itree.contains(pord, point) iff set_itree.contains(point),
        s"$point ${ shouldOrShouldnt(set_itree.contains(point)) } be in interval tree: $set_itree")
      assert(atree.contains(pord, point) iff set_itree.contains(point),
        s"$point ${ shouldOrShouldnt(set_itree.contains(point)) } be in annotation tree: $set_itree")
    }
  }

  @Test def intervalTreeOverlaps() {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      val interval = set_interval.interval
      assert(itree.probablyOverlaps(pord, interval) iff set_itree.probablyOverlaps(set_interval),
        s"$interval ${ shouldOrShouldnt(set_itree.probablyOverlaps(set_interval)) } overlap interval tree: $set_itree")
      assert(atree.probablyOverlaps(pord, interval) iff set_itree.probablyOverlaps(set_interval),
        s"$interval ${ shouldOrShouldnt(set_itree.probablyOverlaps(set_interval)) } overlap annotation tree: $set_itree")
    }
  }

  @Test def intervalTreeEmpty() {
    for (set_itree <- test_itrees) {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      assert(itree.definitelyEmpty(pord) iff set_itree.definitelyEmpty(),
        s"$set_itree ${ shouldOrShouldnt(set_itree.definitelyEmpty()) } be empty")
      assert(atree.definitelyEmpty(pord) iff set_itree.definitelyEmpty(),
        s"$set_itree ${ shouldOrShouldnt(set_itree.definitelyEmpty()) } be empty")
    }
  }

  @Test def intervalTreeDisjoint() {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      val interval = set_interval.interval
      assert(itree.definitelyDisjoint(pord, interval) iff set_itree.definitelyDisjoint(set_interval),
        s"$interval ${ shouldOrShouldnt(set_itree.definitelyDisjoint(set_interval)) } be disjoint from intervals: $set_itree")
      assert(atree.definitelyDisjoint(pord, interval) iff set_itree.definitelyDisjoint(set_interval),
        s"$interval ${ shouldOrShouldnt(set_itree.definitelyDisjoint(set_interval)) } be disjoint from intervals: $set_itree")
    }
  }

  @Test def intervalTreeQueryIntervals() {
    for {
      set_itree <- test_itrees
      point <- points
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      val resulta = atree.queryIntervals(pord, point)
      val resulti = itree.queryIntervals(pord, point)

      assert(resulti.length < 2, s"$set_itree contains multiple queried intervals for $point.\n")
      assert(resulta.toSet == set_itree.queryIntervals(point), s"$set_itree contains nonmatching queried intervals for $point.")
    }
  }

  @Test def intervalTreeQueryValues() {
    for {
      set_itree <- test_itrees
      point <- points
    } yield {
      val itree = set_itree.annotationTree
      val result = itree.queryValues(pord, point)
      assert(result.areDistinct(), s"$set_itree found duplicated queried values for $point.")
      assert(result.toSet == set_itree.queryValues(point), s"$set_itree contains nonmatching queried values for $point.")
    }
  }

  @Test def intervalTreeQueryOverlappingValues() {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val itree = set_itree.annotationTree
      val interval = set_interval.interval
      val result = itree.queryOverlappingValues(pord, interval)
      assert(result.areDistinct(), s"$set_itree found duplicated overlapping values.")
      assert(result.toSet == set_itree.queryProbablyOverlappingValues(set_interval), s"$set_itree contains nonmatching overlapping values for $interval.")
    }
  }

}

case class SetInterval(start: Int, end: Int, includeStart: Boolean, includeEnd: Boolean) {

  val doublepointset: Set[Int] = {
    val first = if (includeStart) 2 * start else 2 * start + 1
    val last = if (includeEnd) 2 * end else 2 * end - 1
    (first to last).toSet
  }

  val interval: Interval = Interval(start, end, includeStart, includeEnd)

  def contains(point: Int): Boolean = doublepointset.contains(2 * point)

  def probablyOverlaps(other: SetInterval): Boolean = doublepointset.intersect(other.doublepointset).nonEmpty

  def definitelyEmpty(): Boolean = doublepointset.isEmpty

  def definitelyDisjoint(other: SetInterval): Boolean = doublepointset.intersect(other.doublepointset).isEmpty
}

case class SetIntervalTree(annotations: Array[(SetInterval, Int)]) {

  val pord: ExtendedOrdering = TInt32().ordering

  val doublepointset: Set[Int] =
    annotations.foldLeft(Set.empty[Int]) { case (ps, (i, _)) => ps.union(i.doublepointset) }

  val (intervals, values) = annotations.unzip

  val annotationTree: IntervalTree[Int] = IntervalTree.annotationTree(pord, annotations.map { case (i, a) => (i.interval, a) } )

  val intervalTree: IntervalTree[Unit] = IntervalTree(pord, intervals.map(_.interval))

  def contains(point: Int): Boolean = doublepointset.contains(2 * point)

  def probablyOverlaps(other: SetInterval): Boolean = doublepointset.intersect(other.doublepointset).nonEmpty

  def definitelyEmpty(): Boolean = doublepointset.isEmpty

  def definitelyDisjoint(other: SetInterval): Boolean = doublepointset.intersect(other.doublepointset).isEmpty

  def queryIntervals(point: Int): Set[Interval] = intervals.filter(_.contains(point)).map(_.interval).toSet

  def queryValues(point: Int): Set[Int] = annotations.filter(_._1.contains(point)).map(_._2).toSet

  def queryProbablyOverlappingValues(interval: SetInterval): Set[Int] = annotations.filter(_._1.probablyOverlaps(interval)).map(_._2).toSet

  override val toString: String = intervals.map(_.interval).mkString(", ")


}