package is.hail.utils

import scala.collection.mutable

class GraphSuite extends munit.FunSuite {

  import Graph._

  private def isIndependentSet[T](x: IndexedSeq[T], g: T => mutable.Set[T]): Boolean = {
    val s = x.toSet
    x.forall(x => g(x).intersect(s).isEmpty)
  }

  test("simple") {
    {
      val actual = maximalIndependentSet(Array((0 -> 1)))
      assert(actual.toSet == Set(0) || actual.toSet == Set(1))
    }

    {
      val actual = maximalIndependentSet(Array(0 -> 1, 0 -> 2))
      assertEquals(actual.toSet, Set(1, 2))
    }

    {
      val actual = maximalIndependentSet(Array(0 -> 1, 0 -> 2, 3 -> 1, 3 -> 2))
      assert(actual.toSet == Set(1, 2) || actual.toSet == Set(0, 3))
    }

    {
      val g = mkGraph(0 -> 1, 0 -> 2, 3 -> 1, 3 -> 2, 3 -> 4)
      val actual = maximalIndependentSet(g)
      assert(isIndependentSet(actual, g))
    }
  }

  test("longCycle") {
    val g = mkGraph(0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 6 -> 0)
    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assertEquals(actual.length, 3)
  }

  test("twoPopularNodes") {
    val g = mkGraph(0 -> 1, 0 -> 2, 0 -> 3, 4 -> 5, 4 -> 6, 4 -> 0)
    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assertEquals(actual.length, 5)
  }

  test("totallyDisconnected") {
    val expected = 0 until 10

    val m = mutable.HashMap.empty[Int, mutable.Set[Int]]
    for (i <- expected)
      m.put(i, mutable.Set())

    val actual = maximalIndependentSet(m)

    assertEquals(actual.toSet, expected.toSet)
  }

  test("disconnected") {
    val g = mkGraph(for (i <- 0 until 10) yield (i, i + 10))

    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assertEquals(actual.length, 10)
  }

  test("selfEdge") {
    val g = mkGraph(0 -> 0, 1 -> 2, 1 -> 3)

    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assertEquals(actual.toSet, Set(2, 3))
  }

  test("emptyGraph") {
    val g = mkGraph[Int]()

    val actual = maximalIndependentSet(g)

    assertEquals(actual.toSeq, Seq.empty[Int])
  }

  test("tieBreakingOfBipartiteGraphWorks") {
    val g = mkGraph(for (i <- 0 until 10) yield (i, i + 10))
    // prefer to remove big numbers
    val actual = maximalIndependentSet(g, Some((l: Int, r: Int) => (l - r).toDouble))

    assert(isIndependentSet(actual, g))
    assertEquals(actual.length, 10)
    assert(actual.forall(_ < 10))
  }

  test("tieBreakingInLongCycleWorks") {
    val g = mkGraph(0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 6 -> 0)
    // prefers to remove small numbers
    val actual = maximalIndependentSet(g, Some((l: Int, r: Int) => (r - l).toDouble))

    assert(isIndependentSet(actual, g))
    assertEquals(actual.length, 3)
    assertEquals(actual.toSet, Set(1, 3, 6))
  }
}
