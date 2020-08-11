package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.asm4s.Code
import is.hail.expr.ir.functions.{IRRandomness, RegistryFunctions}
import is.hail.types.physical.{PInt32, PInt64}
import is.hail.types.virtual.{TArray, TFloat64, TInt32, TInt64, TStream}
import is.hail.utils._
import is.hail.{ExecStrategy, HailContext, HailSuite}
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
  def getTestRNG(mb: EmitMethodBuilder[_], seed: Long): Code[TestIRRandomness] = {
    val rng = mb.genFieldThisRef[IRRandomness]()
    mb.ecb.rngs += rng -> Code.checkcast[IRRandomness](Code.newInstance[TestIRRandomness, Long](seed))
    Code.checkcast[TestIRRandomness](rng)
  }

  def registerAll() {
    registerSeeded0("counter_seeded", TInt32, PInt32(true)) { case (r, rt, seed) =>
      getTestRNG(r.mb, seed).invoke[Int]("counter")
    }

    registerSeeded0("seed_seeded", TInt64, PInt64(true)) { case (r, rt, seed) =>
      getTestRNG(r.mb, seed).invoke[Long]("seed")
    }

    registerSeeded0("pi_seeded", TInt32, PInt32(true)) { case (r, rt, seed) =>
      getTestRNG(r.mb, seed).invoke[Int]("partitionIndex")
    }
  }
}

class RandomFunctionsSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  def counter = ApplySeeded("counter_seeded", FastSeq(), 0L, TInt32)
  val partitionIdx = ApplySeeded("pi_seeded", FastSeq(), 0L, TInt32)

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

    ExecuteContext.scoped() { ctx =>
      val repartitioned = mapped.repartition(ctx, newPartitioner)
      val cachedAndRepartitioned = mapped.cache(ctx).repartition(ctx, newPartitioner)

      assert(mapped.toRows.collect() sameElements repartitioned.toRows.collect())
      assert(mapped.toRows.collect() sameElements cachedAndRepartitioned.toRows.collect())
    }
  }

  @Test def testInterpretIncrementsCorrectly() {
    assertEvalsTo(
      ToArray(StreamMap(StreamRange(0, 3, 1), "i", counter * counter)),
      FastIndexedSeq(0, 1, 4))

    assertEvalsTo(
      StreamFold(StreamRange(0, 3, 1), -1, "j", "i", counter + counter),
      4)

    assertEvalsTo(
      ToArray(StreamFilter(StreamRange(0, 3, 1), "i", Ref("i", TInt32).ceq(counter) && counter.ceq(counter))),
      FastIndexedSeq(0, 1, 2))

    assertEvalsTo(
      ToArray(StreamFlatMap(StreamRange(0, 3, 1),
        "i",
        MakeStream(FastSeq(counter, counter, counter), TStream(TInt32)))),
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
    val actual = CompileAndEvaluate[IndexedSeq[Row]](ctx, GetField(collect(tir), "rows"), false)

    assert(expected.sameElements(actual))
  }

  @Test def testRandCat() {
    val seed = 5L
    assertEvalsTo(invokeSeeded("rand_cat", seed, TInt32, MakeArray(IndexedSeq[IR](0.1), TArray(TFloat64))), 0)
    assertEvalsTo(invokeSeeded("rand_cat", seed, TInt32, MakeArray(IndexedSeq[IR](0.3, 0.2, 0.95, 0.05), TArray(TFloat64))), 1)
    assertEvalsTo(invokeSeeded("rand_cat", seed, TInt32, NA(TArray(TFloat64))), null)
    assertFatal(invokeSeeded("rand_cat", seed, TInt32, MakeArray(IndexedSeq[IR](0.3, NA(TFloat64)), TArray(TFloat64))), "rand_cat")
  }
}
