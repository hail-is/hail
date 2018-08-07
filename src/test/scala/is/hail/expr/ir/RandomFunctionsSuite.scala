package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.asm4s.Code
import is.hail.expr.ir.functions.{IRRandomness, RegistryFunctions}
import is.hail.expr.types._
import is.hail.rvd.{OrderedRVD, OrderedRVDPartitioner}
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class TestIRRandomness(val seed: Long, val partitionIndex: Int) extends IRRandomness(seed, partitionIndex) {
  private[this] var i = -1

  def counter(): Long = {
    i += 1
    i
  }
}

object TestRandomFunctions extends RegistryFunctions {
  var registered = false

  def registerAll() {
    if (!registered) {
      registerSeeded("counter_seeded", TInt64()) { case (mb, seed) =>
        val rand = mb.newLazyField[TestIRRandomness](Code.newInstance[TestIRRandomness, Long, Int](seed, mb.fb.partitionIndexField))
        rand.invoke[Long]("counter")
      }

      registerSeeded("seed_seeded", TInt64()) { case (mb, seed) =>
        val rand = mb.newLazyField[TestIRRandomness](Code.newInstance[TestIRRandomness, Long, Int](seed, mb.fb.partitionIndexField))
        rand.invoke[Long]("seed")
      }

      registerSeeded("pi_seeded", TInt32()) { case (mb, seed) =>
        val rand = mb.newLazyField[TestIRRandomness](Code.newInstance[TestIRRandomness, Long, Int](seed, mb.fb.partitionIndexField))
        rand.invoke[Int]("partitionIndex")
      }
    }
    registered = true
  }
}


class RandomFunctionsSuite extends SparkSuite {

  def mapped2(n: Int, npart: Int) = TableMapRows(
    TableRange(n, npart),
    InsertFields(Ref("row", TableRange(1, 1).typ.rowType),
      FastSeq(
        "seed" -> invoke("seed_seeded", 0L),
        "pi" -> invoke("pi_seeded", 0L),
        "counter" -> invoke("counter_seeded", 0L))),
    Some(FastIndexedSeq("idx")), Some(1))

  @Test def testRandomAcrossJoins() {
    val joined = TableJoin(mapped2(10, 4), mapped2(10, 3), "left")

    val expected = (mapped2(10, 4).execute(hc).rdd.collect(),
    mapped2(10, 3).execute(hc).rdd.collect()).zipped.map { case (Row(idx1, s1, pi1, c1), Row(idx2, s2, pi2, c2)) =>
        assert(idx1 == idx2)
        Row(idx1, s1, s2, pi1, pi2, c1, c2)
    }
    val actual = joined.execute(hc).rdd.collect()

    expected.zip(actual).foreach { case (Row(idx1, s11, s12, pi11, pi12, c11, c12), Row(idx2, s21, s22, pi21, pi22, c21, c22)) =>
      assert(idx1 == idx2)
      assert(s11 == s21, s"$idx1, $s11 vs $s12")
      assert(s12 == s22, s"$idx1, $s12 vs $s22")
      assert(pi11 == pi21, s"$idx1, $pi11 vs $pi12")
      assert(pi12 == pi22, s"$idx1, $pi12 vs $pi22")
      assert(c11 == c21, s"$idx1, $c11 vs $c12")
      assert(c12 == c22, s"$idx1, $c12 vs $c22")

    }
  }

  @Test def testRepartitioning() {
    TestRandomFunctions.registerAll()
    val mappedRVD = mapped2(15, 4).execute(hc).rvd.asInstanceOf[OrderedRVD]

    val newBounds = OrderedRVDPartitioner.makeRangeBoundIntervals(mappedRVD.typ.kType, Array(0, 5, 10, 15).map(Row(_)))
    val newPartitioner = mappedRVD.partitioner.copy(numPartitions = newBounds.length, rangeBounds = newBounds)

    def printrvd(rvd: OrderedRVD): Unit = {
      println("----")
      println(f" ${"idx"}%5s  ${"seed"}%5s  ${"pidx"}%5s  ${"counter"}%7s")
      rvd.toRows.collect().foreach { case Row(idx, seed, pidx, counter) =>
          println(f"$idx%5s  $seed%5s  $pidx%5s   $counter%5s")

      }
      println("----")
    }

    printrvd(mappedRVD)
    printrvd(mappedRVD.constrainToOrderedPartitioner(mappedRVD.typ, newPartitioner))
  }
}
