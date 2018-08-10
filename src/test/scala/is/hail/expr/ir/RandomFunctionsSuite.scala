package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.asm4s.Code
import is.hail.expr.ir.functions.{IRRandomness, RegistryFunctions}
import is.hail.expr.types._
import is.hail.rvd.OrderedRVD
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.{BeforeClass, Test}

class TestIRRandomness(val seed: Long) extends IRRandomness(seed) {
  private[this] var i = -1
  var partitionIndex: Int = 0

  override def reset(pidx: Int) {
    super.reset(pidx)
    partitionIndex = 0
    i = -1
  }

  def counter(): Long = {
    i += 1
    i
  }
}

object TestRandomFunctions extends RegistryFunctions {
  def getTestRNG(mb: EmitMethodBuilder, seed: Long): Code[TestIRRandomness] = {
    val rng = mb.newField[IRRandomness]
    mb.fb.rngs += rng -> Code.checkcast[IRRandomness](Code.newInstance[TestIRRandomness, Long](seed))
    Code.checkcast[TestIRRandomness](rng)
  }

  def registerAll() {
    registerSeeded("counter_seeded", TInt64()) { case (mb, seed) =>
      getTestRNG(mb, seed).invoke[Long]("counter")
    }

    registerSeeded("seed_seeded", TInt64()) { case (mb, seed) =>
      getTestRNG(mb, seed).invoke[Long]("seed")
    }

    registerSeeded("pi_seeded", TInt32()) { case (mb, seed) =>
      getTestRNG(mb, seed).invoke[Int]("partitionIndex")
    }
  }
}

class RandomFunctionsSuite extends SparkSuite {

  def mapped2(n: Int, npart: Int) = TableMapRows(
    TableRange(n, npart),
    InsertFields(Ref("row", TableRange(1, 1).typ.rowType),
      FastSeq(
        "pi" -> ApplySeeded("pi_seeded", FastSeq(), 0L),
        "counter" -> ApplySeeded("counter_seeded", FastSeq(), 0L))),
    Some(FastIndexedSeq("idx")), Some(1))

  @BeforeClass def registerFunctions() {
    TestRandomFunctions.registerAll()
  }

  @Test def testRandomAcrossJoins() {
    def asArray(ir: TableIR) = ir.execute(hc).rdd.collect()

    val joined = TableJoin(mapped2(10, 4), mapped2(10, 3), "left")

    val expected = asArray(mapped2(10, 4)).zip(asArray(mapped2(10, 3)))
      .map { case (Row(idx1, pi1, c1), Row(idx2, pi2, c2)) =>
      assert(idx1 == idx2)
      Row(idx1, pi1, c1, pi2, c2)
    }

    assert(asArray(joined) sameElements expected)
  }

  @Test def testRepartitioning() {
    val mapped = mapped2(15, 4).execute(hc).rvd.asInstanceOf[OrderedRVD]
    val newRangeBounds = FastIndexedSeq(
      Interval(Row(0), Row(4), true, true),
      Interval(Row(4), Row(10), false, true),
      Interval(Row(10), Row(14), false, true))
    val newPartitioner = mapped.partitioner.copy(numPartitions=newRangeBounds.length, rangeBounds=newRangeBounds)

    val repartitioned = mapped.constrainToOrderedPartitioner(mapped.typ, newPartitioner)
    val cachedAndRepartitioned = mapped.cache().constrainToOrderedPartitioner(mapped.typ, newPartitioner)

    assert(mapped.toRows.collect() sameElements repartitioned.toRows.collect())
    assert(mapped.toRows.collect() sameElements cachedAndRepartitioned.toRows.collect())
  }
}