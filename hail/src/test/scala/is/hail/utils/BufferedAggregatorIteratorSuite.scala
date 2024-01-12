package is.hail.utils

import is.hail.check.{Gen, Prop}

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class SumAgg() {
  var x = 0L

  def add(element: Int): Unit =
    x += element

  def comb(other: SumAgg): SumAgg = {
    x += other.x
    this
  }

  override def toString: String = s"SumAgg($x)"
}

class BufferedAggregatorIteratorSuite extends TestNGSuite {
  @Test def test() {
    Prop.forAll(
      Gen.zip(
        Gen.buildableOf[IndexedSeq](Gen.zip(Gen.choose(1, 5), Gen.choose(1, 10))),
        Gen.choose(1, 5),
      )
    ) { case (arr, bufferSize) =>
      val simple = arr.groupBy(_._1).map { case (k, a) => k -> a.map(_._2.toLong).sum }
      val buffAgg = {
        new BufferedAggregatorIterator[(Int, Int), SumAgg, SumAgg, Int](
          arr.iterator,
          () => new SumAgg(),
          { case (k, v) => k },
          { case (t, agg) => agg.add(t._2) },
          a => a,
          bufferSize,
        )
          .toArray
          .groupBy(_._1)
          .mapValues(sums => sums.map(_._2).fold(new SumAgg()) { case (s1, s2) => s1.comb(s2) }.x)
      }
      simple == buffAgg
    }.check()
  }
}
