package is.hail.annotations

import is.hail.expr.ir.agg._

class ApproxCDFAggregatorSuite extends munit.FunSuite {
  test("merge") {
    val array: Array[Double] = Array(1, 3, 5, 0, 0, 0, 2, 4, 6)
    ApproxCDFHelper.merge(array, 0, 3, array, 6, 9, array, 3)
    assert(array.view.slice(3, 9) sameElements Range(1, 7))
  }

  test("compact level zero") {
    val rand = new java.util.Random(1) // first Boolean is `true`
    val levels: Array[Int] = Array(0, 4, 7, 10)
    val items: Array[Double] = Array(7, 2, 6, 4, 1, 3, 8, 0, 5, 9)
    val compactionCounts: Array[Int] = Array(0, 0, 0)
    val combiner = new ApproxCDFCombiner(levels, items, compactionCounts, 3, rand)
    combiner.compactLevel(0): Unit
    assert(items.view.slice(1, 10) sameElements Array(2, 7, 1, 3, 6, 8, 0, 5, 9))
  }

  test("compact level") {
    val rand = new java.util.Random(1) // first Boolean is `true`
    val levels: Array[Int] = Array(0, 3, 6, 9)
    val items: Array[Double] = Array(7, 2, 4, 1, 3, 8, 0, 5, 9)
    val compactionCounts: Array[Int] = Array(0, 0, 0)
    val combiner = new ApproxCDFCombiner(levels, items, compactionCounts, 3, rand)
    combiner.compactLevel(1): Unit
    assert(items.view.slice(1, 9) sameElements Array(7, 2, 4, 1, 0, 5, 8, 9))
  }
}
