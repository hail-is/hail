package is.hail.utils

import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.TInt32
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import org.testng.Assert._

class IntervalSuite extends TestNGSuite {

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
    SetIntervalTree(Array[(SetInterval, Int)]()) +:
      test_intervals.flatMap { i1 =>
        SetIntervalTree(Array(i1).zipWithIndex) +:
          test_intervals.map { i2 =>
            SetIntervalTree(Array(i1, i2).zipWithIndex)
          }
      } :+ SetIntervalTree(test_intervals.toArray.zipWithIndex)


  @Test def interval_agrees_with_set_interval_definitely_empty() {
    for (set_interval <- test_intervals) {
      val interval = set_interval.interval
      assertEquals(interval.definitelyEmpty(pord), set_interval.definitelyEmpty())
    }
  }

  @Test def interval_agrees_with_set_interval_greater_than_point() {
    for (set_interval <- test_intervals; p <- points) {
      val interval = set_interval.interval
      assertEquals(interval.isAbovePosition(pord, p), set_interval.doubledPointSet.forall(dp => dp > 2 * p))
    }
  }

  @Test def interval_agrees_with_set_interval_less_than_point() {
    for (set_interval <- test_intervals; p <- points) {
      val interval = set_interval.interval
      assertEquals(interval.isBelowPosition(pord, p), set_interval.doubledPointSet.forall(dp => dp < 2 * p))
    }
  }

  @Test def interval_agrees_with_set_interval_contains() {
    for (set_interval <- test_intervals; p <- points) {
      val interval = set_interval.interval
      assertEquals(interval.contains(pord, p), set_interval.contains(p))
    }
  }

  @Test def interval_agrees_with_set_interval_includes() {
    for (set_interval1 <- test_intervals; set_interval2 <- test_intervals) {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.includes(pord, interval2), set_interval1.includes(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_probably_overlaps() {
    for (set_interval1 <- test_intervals; set_interval2 <- test_intervals) {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.mayOverlap(pord, interval2), set_interval1.probablyOverlaps(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_definitely_disjoint() {
    for (set_interval1 <- test_intervals; set_interval2 <- test_intervals) {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.definitelyDisjoint(pord, interval2), set_interval1.definitelyDisjoint(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_disjoint_greater_than() {
    for {set_interval1 <- test_intervals
    set_interval2 <- test_intervals} {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.isAbove(pord, interval2), set_interval1.isAboveInterval(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_disjoint_less_than() {
    for {set_interval1 <- test_intervals
    set_interval2 <- test_intervals} {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.isBelow(pord, interval2), set_interval1.isBelowInterval(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_mergeable() {
    for {set_interval1 <- test_intervals
    set_interval2 <- test_intervals} {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.canMergeWith(pord, interval2), set_interval1.mergeable(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_merge() {
    for {set_interval1 <- test_intervals
    set_interval2 <- test_intervals} {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.merge(pord, interval2), set_interval1.union(set_interval2).map(_.interval))
    }
  }

  @Test def interval_agrees_with_set_interval_intersect() {
    for (set_interval1 <- test_intervals; set_interval2 <- test_intervals) {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.intersect(pord, interval2), set_interval1.intersect(set_interval2).interval)
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_contains() {
    for {
      set_itree <- test_itrees
      p <- points
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      assertEquals(itree.contains(pord, p), set_itree.contains(p))
      assertEquals(atree.contains(pord, p), set_itree.contains(p))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_probably_overlaps() {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      val interval = set_interval.interval
      assertEquals(itree.probablyOverlaps(pord, interval), set_itree.probablyOverlaps(set_interval))
      assertEquals(atree.probablyOverlaps(pord, interval), set_itree.probablyOverlaps(set_interval))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_definitely_empty() {
    for (set_itree <- test_itrees) {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      assertEquals(itree.definitelyEmpty(pord), set_itree.definitelyEmpty())
      assertEquals(atree.definitelyEmpty(pord), set_itree.definitelyEmpty())
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_definitely_disjoint() {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      val interval = set_interval.interval
      assertEquals(itree.definitelyDisjoint(pord, interval), set_itree.definitelyDisjoint(set_interval))
      assertEquals(atree.definitelyDisjoint(pord, interval), set_itree.definitelyDisjoint(set_interval))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_query_intervals() {
    for {
      set_itree <- test_itrees
      point <- points
    } yield {
      val atree = set_itree.annotationTree
      val itree = set_itree.intervalTree
      val resulta = atree.queryIntervals(pord, point)
      val resulti = itree.queryIntervals(pord, point)

      assertTrue(resulti.length < 2)
      assertEquals(resulta.toSet, set_itree.queryIntervals(point))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_query_values() {
    for {
      set_itree <- test_itrees
      point <- points
    } yield {
      val itree = set_itree.annotationTree
      val result = itree.queryValues(pord, point)
      assertTrue(result.areDistinct())
      assertEquals(result.toSet, set_itree.queryValues(point))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_query_overlapping_values() {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val itree = set_itree.annotationTree
      val interval = set_interval.interval
      val result = itree.queryOverlappingValues(pord, interval)
      assertTrue(result.areDistinct())
      assertEquals(result.toSet, set_itree.queryProbablyOverlappingValues(set_interval))
    }
  }
}

object SetInterval {
  def from(i: Interval): SetInterval =
    SetInterval(i.start.asInstanceOf[Int], i.end.asInstanceOf[Int], i.includesStart, i.includesEnd)
}

case class SetInterval(start: Int, end: Int, includesStart: Boolean, includesEnd: Boolean) {

  val pord: ExtendedOrdering = TInt32().ordering

  val doubledPointSet: Set[Int] = {
    val first = if (includesStart) 2 * start else 2 * start + 1
    val last = if (includesEnd) 2 * end else 2 * end - 1
    (first to last).toSet
  }

  val interval: Interval = Interval(start, end, includesStart, includesEnd)

  def contains(point: Int): Boolean = doubledPointSet.contains(2 * point)

  def includes(other: SetInterval): Boolean =
    (other.doubledPointSet -- this.doubledPointSet).isEmpty

  def probablyOverlaps(other: SetInterval): Boolean = doubledPointSet.intersect(other.doubledPointSet).nonEmpty

  def definitelyEmpty(): Boolean = doubledPointSet.isEmpty

  def definitelyDisjoint(other: SetInterval): Boolean = doubledPointSet.intersect(other.doubledPointSet).isEmpty

  def isAboveInterval(other: SetInterval): Boolean =
    doubledPointSet.forall(p1 => other.doubledPointSet.forall(p2 => p1 > p2))

  def isBelowInterval(other: SetInterval): Boolean =
    doubledPointSet.forall(p1 => other.doubledPointSet.forall(p2 => p1 < p2))

  def mergeable(other: SetInterval): Boolean = {
    val combinedPoints = doubledPointSet.union(other.doubledPointSet)
    if (combinedPoints.isEmpty)
      true
    else {
      val start = combinedPoints.min(pord.toOrdering)
      val end = combinedPoints.max(pord.toOrdering)
      (start to end).forall(combinedPoints.contains)
    }
  }

  def unionedPoints(other: SetInterval): Set[Int] = doubledPointSet.union(other.doubledPointSet)

  def union(other: SetInterval): Option[SetInterval] = {
    val combined = doubledPointSet.union(other.doubledPointSet)
    if (combined.isEmpty)
      return Some(this)
    if (mergeable(other)) {
      val start = combined.min(pord.toOrdering)
      val end = combined.max(pord.toOrdering)
      Some(SetInterval(start / 2, (end + 1) / 2, start % 2 == 0, end % 2 == 0))
    }
    else None
  }

  def intersect(other: SetInterval): SetInterval = {
    val intersection = doubledPointSet.intersect(other.doubledPointSet)
    if (intersection.isEmpty)
      SetInterval(start, start, false, false)
    else {
      val start = intersection.min(pord.toOrdering)
      val end = intersection.max(pord.toOrdering)
      SetInterval(start / 2, (end + 1) / 2, start % 2 == 0, end % 2 == 0)
    }
  }
}

case class SetIntervalTree(annotations: Array[(SetInterval, Int)]) {

  val pord: ExtendedOrdering = TInt32().ordering

  val doubledPointSet: Set[Int] =
    annotations.foldLeft(Set.empty[Int]) { case (ps, (i, _)) => ps.union(i.doubledPointSet) }

  val (intervals, values) = annotations.unzip

  val annotationTree: IntervalTree[Int] = IntervalTree.annotationTree(pord, annotations.map { case (i, a) => (i.interval, a) })

  val intervalTree: IntervalTree[Unit] = IntervalTree(pord, intervals.map(_.interval))

  def contains(point: Int): Boolean = doubledPointSet.contains(2 * point)

  def probablyOverlaps(other: SetInterval): Boolean = doubledPointSet.intersect(other.doubledPointSet).nonEmpty

  def definitelyEmpty(): Boolean = doubledPointSet.isEmpty

  def definitelyDisjoint(other: SetInterval): Boolean = doubledPointSet.intersect(other.doubledPointSet).isEmpty

  def queryIntervals(point: Int): Set[Interval] = intervals.filter(_.contains(point)).map(_.interval).toSet

  def queryValues(point: Int): Set[Int] = annotations.filter(_._1.contains(point)).map(_._2).toSet

  def queryProbablyOverlappingValues(interval: SetInterval): Set[Int] = annotations.filter(_._1.probablyOverlaps(interval)).map(_._2).toSet

  override val toString: String = intervals.map(_.interval).mkString(", ")
}