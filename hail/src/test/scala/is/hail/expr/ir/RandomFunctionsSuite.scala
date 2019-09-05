package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.asm4s.Code
import is.hail.expr.ir.functions.{IRRandomness, RegistryFunctions}
import is.hail.expr.types._
import is.hail.rvd.RVD
import is.hail.TestUtils._
import is.hail.expr.types.virtual.{TArray, TInt32, TInt64}
import is.hail.table.Table
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

  def counter(): Int = {
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
    registerSeeded("counter_seeded", TInt32(), null) { case (r, rt, seed) =>
      getTestRNG(r.mb, seed).invoke[Int]("counter")
    }

    registerSeeded("seed_seeded", TInt64(), null) { case (r, rt, seed) =>
      getTestRNG(r.mb, seed).invoke[Long]("seed")
    }

    registerSeeded("pi_seeded", TInt32(), null) { case (r, rt, seed) =>
      getTestRNG(r.mb, seed).invoke[Int]("partitionIndex")
    }
  }
}

class RandomFunctionsSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  val counter = ApplySeeded("counter_seeded", FastSeq(), 0L, TInt32())
  val partitionIdx = ApplySeeded("pi_seeded", FastSeq(), 0L, TInt32())

  def mapped2(n: Int, npart: Int) = TableMapRows(
    TableRange(n, npart),
    InsertFields(Ref("row", TableRange(1, 1).typ.rowType),
      FastSeq(
        "pi" -> partitionIdx,
        "counter" -> counter)))

  @BeforeClass def registerFunctions() {
    TestRandomFunctions.registerAll()
  }

  @Test def testRandomAcrossJoins() {
    def asArray(ir: TableIR) = Interpret(ir, ctx).rdd.collect()

    val joined = TableJoin(
      mapped2(10, 4),
      TableRename(mapped2(10, 3), Map("pi" -> "pi2", "counter" -> "counter2"), Map.empty),
      "left")

    val expected = asArray(mapped2(10, 4)).zip(asArray(mapped2(10, 3)))
      .map { case (Row(idx1, pi1, c1), Row(idx2, pi2, c2)) =>
      assert(idx1 == idx2)
      Row(idx1, pi1, c1, pi2, c2)
    }

    assert(asArray(joined) sameElements expected)
  }

  @Test def testRepartitioningAfterRandomness() {
    val mapped = Interpret(mapped2(15, 4), ctx).rvd
    val newRangeBounds = FastIndexedSeq(
      Interval(Row(0), Row(4), true, true),
      Interval(Row(4), Row(10), false, true),
      Interval(Row(10), Row(14), false, true))
    val newPartitioner = mapped.partitioner.copy(rangeBounds=newRangeBounds)

    val repartitioned = mapped.repartition(newPartitioner)
    val cachedAndRepartitioned = mapped.cache().repartition(newPartitioner)

    assert(mapped.toRows.collect() sameElements repartitioned.toRows.collect())
    assert(mapped.toRows.collect() sameElements cachedAndRepartitioned.toRows.collect())
  }

  @Test def testInterpretIncrementsCorrectly() {
    assertEvalsTo(
      ArrayMap(ArrayRange(0, 3, 1), "i", counter * counter),
      FastIndexedSeq(0, 1, 4))

    assertEvalsTo(
      ArrayFold(ArrayRange(0, 3, 1), -1, "j", "i", counter + counter),
      4)

    assertEvalsTo(
      ArrayFilter(ArrayRange(0, 3, 1), "i", Ref("i", TInt32()).ceq(counter) && counter.ceq(counter)),
      FastIndexedSeq(0, 1, 2))

    assertEvalsTo(
      ArrayFlatMap(ArrayRange(0, 3, 1),
        "i",
        MakeArray(FastSeq(counter, counter, counter), TArray(TInt32()))),
      FastIndexedSeq(0, 0, 0, 1, 1, 1, 2, 2, 2))
  }

  @Test def testRepartitioningSimplifyRules() {
    val tir =
    TableMapRows(
      TableHead(
        TableMapRows(
          TableRange(10, 3),
          Ref("row", TableRange(1, 1).typ.rowType)),
        5L),
      InsertFields(
        Ref("row", TableRange(1, 1).typ.rowType),
        FastSeq(
          "pi" -> partitionIdx,
          "counter" -> counter)))

    val expected = Interpret(tir, ctx).rvd.toRows.collect()
    val actual = new Table(hc, tir).rdd.collect()

    assert(expected.sameElements(actual))
  }
}
