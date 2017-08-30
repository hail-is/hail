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
      println(actual: IndexedSeq[Int])
      assert(isIndependentSet(actual, g))
    }
  }

  @Test def longCycle() {
    val g = mkGraph(0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 6 -> 0)
    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assert(actual.length > 1)
  }

  @Test def twoPopularNodes() {
    val g = mkGraph(0 -> 1, 0 -> 2, 0 -> 3, 4 -> 5, 4 -> 6, 4 -> 0)
    val actual = maximalIndependentSet(g)

    assert(isIndependentSet(actual, g))
    assert(actual.length == 5)
  }
}
