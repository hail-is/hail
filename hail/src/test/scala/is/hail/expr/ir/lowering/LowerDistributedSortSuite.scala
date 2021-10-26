package is.hail.expr.ir.lowering

import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.ir.functions.IRRandomness
import is.hail.expr.ir.{Ascending, IR, Literal, Requiredness, RequirednessAnalysis, SortField, TableRange, ToArray, ToStream}
import is.hail.{ExecStrategy, HailSuite, TestUtils}
import is.hail.expr.ir.lowering.LowerDistributedSort.samplePartition
import is.hail.types.RTable
import is.hail.types.virtual.{TArray, TInt32, TStruct}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LowerDistributedSortSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.compileOnly

  @Test def testSamplePartition() {
    val dataKeys = IndexedSeq((0, 0), (0, -1), (1, 4), (2, 8), (3, 4), (4, 5), (5, 3), (6, 9), (7, 7), (8, -3), (9, 1))
    val elementType = TStruct(("key1", TInt32), ("key2", TInt32), ("value", TInt32))
    val data1 = ToStream(Literal(TArray(elementType), dataKeys.map{ case (k1, k2) => Row(k1, k2, k1 * k1)}))
    val sampleSeq = ToStream(Literal(TArray(TInt32), IndexedSeq(0, 2, 3, 7)))

    val sampled = samplePartition(data1, sampleSeq, IndexedSeq("key1", "key2"))

    assertEvalsTo(sampled, Row(Row(0, -1), Row(9, 1), IndexedSeq( Row(0, 0, 0), Row(1, 4, 1), Row(2, 8, 4), Row(6, 9, 36))))
  }

  @Test def testHowManyPerPartition(): Unit = {
    val partitionCounts = (0 until 10).map(idx => 100)
    val rand = new IRRandomness((math.random() * 1000).toInt)
    println(LowerDistributedSort.howManySamplesPerPartition(rand, 1000, 10, partitionCounts))
  }

  @Test def testDistributedSort(): Unit = {
    val myTable = TableRange(100, 10)
    val req: RequirednessAnalysis = Requiredness(myTable, ctx)
    val rowType = req.lookup(myTable).asInstanceOf[RTable].rowType
    val stage = LowerTableIR.applyTable(myTable, DArrayLowering.All, ctx, req, Map.empty[String, IR])
    val sortFields = IndexedSeq[SortField](SortField("idx", Ascending))
    println(TestUtils.eval(LowerDistributedSort.distributedSort(ctx, stage, sortFields, Map.empty[String, IR], rowType)))
    //assertEvalsTo(LowerDistributedSort.distributedSort(ctx, stage, sortFields, Map.empty[String, IR], rowType), null)
  }
}
