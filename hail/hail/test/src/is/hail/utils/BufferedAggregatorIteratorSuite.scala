package is.hail.utils

import org.scalacheck.Gen
import org.scalacheck.Gen._
import org.scalatest.matchers.should.Matchers.{be, convertToAnyShouldWrapper}
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

class SumAgg {
  var x: Long = 0

  def seq(element: Long): Unit =
    x += element

  def comb(other: SumAgg): SumAgg = { x += other.x; this }

  override def toString: String =
    s"${getClass.getSimpleName}($x)"
}

class BufferedAggregatorIteratorSuite extends TestNGSuite with ScalaCheckDrivenPropertyChecks {

  private[this] lazy val gen: Gen[(Array[(Int, Long)], Int)] =
    for {
      data <- containerOf[Array, (Int, Long)](zip(choose(1, 5), choose(1L, 10L)))
      len <- choose(1, 5)
    } yield (data, len)

  @Test def test(): Unit =
    forAll(gen) { case (arr, bufferSize) =>
      val simple: Map[Int, Long] =
        arr.groupBy(_._1).map { case (k, a) => k -> a.map(_._2).sum }

      val buffAgg: Map[Int, Long] =
        new BufferedAggregatorIterator[(Int, Long), SumAgg, SumAgg, Int](
          arr.iterator,
          () => new SumAgg,
          { case (k, _) => k },
          { case (t, agg) => agg.seq(t._2) },
          a => a,
          bufferSize,
        )
          .toArray
          .groupBy(_._1)
          .mapValues(_.map(_._2).fold(new SumAgg()) { case (s1, s2) => s1.comb(s2) }.x)

      simple should be(buffAgg)
    }

}
