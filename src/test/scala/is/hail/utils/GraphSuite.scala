package is.hail.utils

import org.testng.annotations.Test
import org.scalatest._
import Matchers._

import scala.collection.mutable
import scala.reflect.ClassTag

class GraphSuite {
  import Graph._

  private def isIndependentSet[T](x: Array[T], g: T => mutable.Set[T]): Boolean = {
    val s = x.toSet
    x.forall(x => g(x).intersect(s).isEmpty)
  }

  @Test def simple() {
    {
      val actual = maximalIndependentSet(Array((0 -> 1)))
      actual should ((contain theSameElementsAs Array(0)) or (contain theSameElementsAs Array(1)))
    }

    {
      val actual = maximalIndependentSet(Array(0 -> 1, 0 -> 2))
      actual should contain theSameElementsAs Array(1,2)
    }

    {
      val actual = maximalIndependentSet(Array(0 -> 1, 0 -> 2, 3 -> 1, 3 -> 2))
      actual should ((contain theSameElementsAs Array(1,2)) or (contain theSameElementsAs Array(0,3)))
    }

    {
      val g = mkGraph(0 -> 1, 0 -> 2, 3 -> 1, 3 -> 2, 3 -> 4)
      val actual = maximalIndependentSet(g)
      assert(isIndependentSet(actual, g))
    }
  }

  @Test def longCycle() {
    val g = mkGraph(0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 6 -> 0)
    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assert(actual.length == 3)
  }

  @Test def twoPopularNodes() {
    val g = mkGraph(0 -> 1, 0 -> 2, 0 -> 3, 4 -> 5, 4 -> 6, 4 -> 0)
    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assert(actual.length == 5)
  }

  @Test def totallyDisconnected() {
    val expected = 0 until 10

    val m = new mutable.HashMap[Int, mutable.Set[Int]]() with mutable.MultiMap[Int, Int]
    for (i <- expected) {
      m.put(i, mutable.Set())
    }

    val actual = maximalIndependentSet(m)

    actual should contain theSameElementsAs expected
  }

  @Test def disconnected() {
    val g = mkGraph(for (i <- 0 until 10) yield (i, i + 10))

    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assert(actual.length == 10)
  }

  @Test def selfEdge() {
    val g = mkGraph(0 -> 0, 1 -> 2, 1 -> 3)

    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    actual should contain theSameElementsAs Array(2,3)
  }

  @Test def emptyGraph() {
    val g = mkGraph[Int]()

    val actual = maximalIndependentSet(g)

    assert(actual === Array[Int]())
  }

  @Test def tieBreakingOfBipartiteGraphWorks() {
    val g = mkGraph(for (i <- 0 until 10) yield (i, i + 10))
    // prefer to remove big numbers
    val actual = maximalIndependentSet(g, Some((l: Int, r: Int) => l - r))

    assert(isIndependentSet(actual, g))
    assert(actual.length == 10)
    assert(actual.forall(_ < 10))
  }


  @Test def tieBreakingInLongCycleWorks() {
    val g = mkGraph(0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 6 -> 0)
    // prefers to remove small numbers
    val actual = maximalIndependentSet(g, Some((l: Int, r: Int) => r - l))

    assert(isIndependentSet(actual, g))
    assert(actual.length == 3)
    actual should contain theSameElementsAs Array[Int](1, 3, 6)
  }
}
