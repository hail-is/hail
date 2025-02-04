package is.hail.utils

import is.hail.HailSuite
import is.hail.annotations.ExtendedOrdering
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual.{TInt32, TStruct}

import org.apache.spark.sql.Row
import org.testng.Assert._
import org.testng.ITestContext
import org.testng.annotations.{BeforeMethod, Test}

class IntervalSuite extends HailSuite {

  val pord: ExtendedOrdering = TInt32.ordering(HailStateManager(Map.empty))

  // set of intervals chosen from 5 endpoints spans the space of relations
  // that two non-empty intervals can have with each other.
  val points: IndexedSeq[Int] = 1 to 5

  val test_intervals: IndexedSeq[SetInterval] =
    for {
      s <- points
      e <- points
      is <- Array(true, false)
      ie <- Array(true, false)
      if pord.lt(s, e) || (pord.equiv(s, e) && is && ie)
    } yield SetInterval(s, e, is, ie)

  var test_itrees: IndexedSeq[SetIntervalTree] = _

  @BeforeMethod
  def setupIntervalTrees(context: ITestContext): Unit = {
    test_itrees = SetIntervalTree(ctx, Array[(SetInterval, Int)]()) +:
      test_intervals.flatMap { i1 =>
        SetIntervalTree(ctx, Array(i1).zipWithIndex) +:
          test_intervals.flatMap { i2 =>
            if (i1.end <= i2.start)
              Some(SetIntervalTree(ctx, Array(i1, i2).zipWithIndex))
            else
              None
          }
      }
  }

  @Test def interval_agrees_with_set_interval_greater_than_point(): Unit = {
    for {
      set_interval <- test_intervals
      p <- points
    } {
      val interval = set_interval.interval
      assertEquals(
        interval.isAbovePosition(pord, p),
        set_interval.doubledPointSet.forall(dp => dp > 2 * p),
      )
    }
  }

  @Test def interval_agrees_with_set_interval_less_than_point(): Unit = {
    for {
      set_interval <- test_intervals
      p <- points
    } {
      val interval = set_interval.interval
      assertEquals(
        interval.isBelowPosition(pord, p),
        set_interval.doubledPointSet.forall(dp => dp < 2 * p),
      )
    }
  }

  @Test def interval_agrees_with_set_interval_contains(): Unit = {
    for {
      set_interval <- test_intervals
      p <- points
    } {
      val interval = set_interval.interval
      assertEquals(interval.contains(pord, p), set_interval.contains(p))
    }
  }

  @Test def interval_agrees_with_set_interval_includes(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.includes(pord, interval2), set_interval1.includes(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_probably_overlaps(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(
        interval1.overlaps(pord, interval2),
        set_interval1.probablyOverlaps(set_interval2),
      )
    }
  }

  @Test def interval_agrees_with_set_interval_definitely_disjoint(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(
        interval1.isDisjointFrom(pord, interval2),
        set_interval1.definitelyDisjoint(set_interval2),
      )
    }
  }

  @Test def interval_agrees_with_set_interval_disjoint_greater_than(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.isAbove(pord, interval2), set_interval1.isAboveInterval(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_disjoint_less_than(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.isBelow(pord, interval2), set_interval1.isBelowInterval(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_mergeable(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(interval1.canMergeWith(pord, interval2), set_interval1.mergeable(set_interval2))
    }
  }

  @Test def interval_agrees_with_set_interval_merge(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(
        interval1.merge(pord, interval2),
        set_interval1.union(set_interval2).map(_.interval),
      )
    }
  }

  @Test def interval_agrees_with_set_interval_intersect(): Unit = {
    for {
      set_interval1 <- test_intervals
      set_interval2 <- test_intervals
    } {
      val interval1 = set_interval1.interval
      val interval2 = set_interval2.interval
      assertEquals(
        interval1.intersect(pord, interval2),
        set_interval1.intersect(set_interval2).map(_.interval),
      )
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_contains(): Unit = {
    for {
      set_itree <- test_itrees
      p <- points
    } yield {
      val itree = set_itree.intervalTree
      assertEquals(itree.contains(Row(p)), set_itree.contains(p))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_probably_overlaps(): Unit = {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val itree = set_itree.intervalTree
      val interval = set_interval.rowInterval
      assertEquals(itree.overlaps(interval), set_itree.probablyOverlaps(set_interval))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_definitely_disjoint(): Unit = {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val itree = set_itree.intervalTree
      val interval = set_interval.rowInterval
      assertEquals(itree.isDisjointFrom(interval), set_itree.definitelyDisjoint(set_interval))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_query_values(): Unit = {
    for {
      set_itree <- test_itrees
      point <- points
    } yield {
      val itree = set_itree.intervalTree
      val result = itree.queryKey(Row(point))
      assertTrue(result.areDistinct())
      assertEquals(result.toSet, set_itree.queryValues(point))
    }
  }

  @Test def interval_tree_agrees_with_set_interval_tree_query_overlapping_values(): Unit = {
    for {
      set_itree <- test_itrees
      set_interval <- test_intervals
    } yield {
      val itree = set_itree.intervalTree
      val interval = set_interval.rowInterval
      val result = itree.queryInterval(interval)
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

  val pord: ExtendedOrdering = TInt32.ordering(HailStateManager(Map.empty))

  val doubledPointSet: Set[Int] = {
    val first = if (includesStart) 2 * start else 2 * start + 1
    val last = if (includesEnd) 2 * end else 2 * end - 1
    (first to last).toSet
  }

  val interval: Interval = Interval(start, end, includesStart, includesEnd)

  val rowInterval: Interval = Interval(Row(start), Row(end), includesStart, includesEnd)

  def contains(point: Int): Boolean = doubledPointSet.contains(2 * point)

  def includes(other: SetInterval): Boolean =
    (other.doubledPointSet -- this.doubledPointSet).isEmpty

  def probablyOverlaps(other: SetInterval): Boolean =
    doubledPointSet.intersect(other.doubledPointSet).nonEmpty

  def definitelyEmpty(): Boolean = doubledPointSet.isEmpty

  def definitelyDisjoint(other: SetInterval): Boolean =
    doubledPointSet.intersect(other.doubledPointSet).isEmpty

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
    } else None
  }

  def intersect(other: SetInterval): Option[SetInterval] = {
    val intersection = doubledPointSet.intersect(other.doubledPointSet)
    if (this.definitelyDisjoint(other))
      None
    else {
      assert(intersection.nonEmpty)
      val start = intersection.min(pord.toOrdering)
      val end = intersection.max(pord.toOrdering)
      Some(SetInterval(start / 2, (end + 1) / 2, start % 2 == 0, end % 2 == 0))
    }
  }
}

case class SetIntervalTree(ctx: ExecuteContext, annotations: Array[(SetInterval, Int)]) {

  val pord: ExtendedOrdering = TInt32.ordering(HailStateManager(Map.empty))

  val doubledPointSet: Set[Int] =
    annotations.foldLeft(Set.empty[Int]) { case (ps, (i, _)) => ps.union(i.doubledPointSet) }

  val (intervals, values) = annotations.unzip

  val intervalTree: RVDPartitioner =
    new RVDPartitioner(ctx.stateManager, TStruct(("i", TInt32)), intervals.map(_.rowInterval))

  def contains(point: Int): Boolean = doubledPointSet.contains(2 * point)

  def probablyOverlaps(other: SetInterval): Boolean =
    doubledPointSet.intersect(other.doubledPointSet).nonEmpty

  def definitelyEmpty(): Boolean = doubledPointSet.isEmpty

  def definitelyDisjoint(other: SetInterval): Boolean =
    doubledPointSet.intersect(other.doubledPointSet).isEmpty

  def queryIntervals(point: Int): Set[Interval] =
    intervals.filter(_.contains(point)).map(_.interval).toSet

  def queryValues(point: Int): Set[Int] = annotations.filter(_._1.contains(point)).map(_._2).toSet

  def queryProbablyOverlappingValues(interval: SetInterval): Set[Int] =
    annotations.filter(_._1.probablyOverlaps(interval)).map(_._2).toSet

  override val toString: String = intervals.map(_.interval).mkString(", ")
}
