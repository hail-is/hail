package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.{Annotation, BroadcastIndexedSeq}
import is.hail.check.{Gen, Prop}
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.language._

class ConcordanceSuite extends SparkSuite {

  @Test def testCombiner() {
    val comb = new ConcordanceCombiner

    comb.merge(1, 3)
    comb.merge(1, 3)
    comb.merge(1, 3)
    comb.merge(1, 3)
    comb.merge(0, 4)
    comb.merge(2, 0)

    assert(comb.toAnnotation == FastIndexedSeq(
      FastIndexedSeq(0L, 0L, 0L, 0L, 1L),
      FastIndexedSeq(0L, 0L, 0L, 4L, 0L),
      FastIndexedSeq(1L, 0L, 0L, 0L, 0L),
      FastIndexedSeq(0L, 0L, 0L, 0L, 0L),
      FastIndexedSeq(0L, 0L, 0L, 0L, 0L)
    ))

    val comb2 = new ConcordanceCombiner

    comb2.merge(4, 0)
    comb2.merge(4, 0)
    comb2.merge(1, 0)
    comb2.merge(1, 0)
    comb2.merge(4, 0)
    comb2.merge(0, 2)
    comb2.merge(0, 3)
    comb2.merge(0, 3)
    comb2.merge(3, 3)
    comb2.merge(3, 3)
    comb2.merge(1, 3)
    comb2.merge(1, 3)
    comb2.merge(3, 1)
    comb2.merge(3, 1)
    comb2.merge(4, 1)
    comb2.merge(4, 1)
    comb2.merge(4, 1)

    assert(comb2.toAnnotation == FastIndexedSeq(
      FastIndexedSeq(0L, 0L, 1L, 2L, 0L),
      FastIndexedSeq(2L, 0L, 0L, 2L, 0L),
      FastIndexedSeq(0L, 0L, 0L, 0L, 0L),
      FastIndexedSeq(0L, 2L, 0L, 2L, 0L),
      FastIndexedSeq(3L, 3L, 0L, 0L, 0L)
    ))

    assert(comb2.nDiscordant == 0)
  }

  @Test def testNDiscordant() {
    val g = (for {i <- Gen.choose(-2, 2)
      j <- Gen.choose(-2, 2)} yield (i, j)).filter { case (i, j) => !(i == -2 && j == -2) }
    val seqG = Gen.buildableOf[Array](g)

    val comb = new ConcordanceCombiner

    Prop.forAll(seqG) { values =>
      comb.reset()

      var n = 0
      values.foreach { case (i, j) =>
        if (i == -2)
          comb.merge(0, j + 2)
        else if (j == -2)
          comb.merge(i + 2, 0)
        else {
          if (i >= 0 && j >= 0 && i != j)
            n += 1
          comb.merge(i + 2, j + 2)
        }
      }
      n == comb.nDiscordant
    }.check()
  }
}
