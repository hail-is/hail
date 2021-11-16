package is.hail.expr.ir.lowering

import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.ir.functions.IRRandomness
import is.hail.expr.ir.{Apply, ApplyBinaryPrimOp, Ascending, ErrorIDs, GetField, I32, IR, Literal, MakeStruct, Ref, Requiredness, RequirednessAnalysis, SortField, TableIR, TableMapRows, TableRange, ToArray, ToStream}
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

    assertEvalsTo(sampled, Row(Row(0, -1), Row(9, 1), IndexedSeq( Row(0, 0), Row(1, 4), Row(2, 8), Row(6, 9)), false))

    val dataKeys2 = IndexedSeq((0, 0), (0, 1), (1, 0), (3, 3))
    val elementType2 = TStruct(("key1", TInt32), ("key2", TInt32))
    val data2 = ToStream(Literal(TArray(elementType2), dataKeys2.map{ case (k1, k2) => Row(k1, k2)}))
    val sampleSeq2 = ToStream(Literal(TArray(TInt32), IndexedSeq(0)))
    val sampled2 = samplePartition(data2, sampleSeq2, IndexedSeq("key2", "key1"))
    assertEvalsTo(sampled2, Row(Row(0, 0), Row(3, 3), IndexedSeq( Row(0, 0)), false))
  }

  @Test def testHowManyPerPartition(): Unit = {
    val partitionCounts = (0 until 10).map(idx => 100)
    val rand = new IRRandomness((math.random() * 1000).toInt)
    println(LowerDistributedSort.howManySamplesPerPartition(rand, 1000, 10, partitionCounts))
  }

  // Only does ascending for now
  def testDistributedSortHelper(myTable: TableIR, sortFieldNames: IndexedSeq[String]): Unit = {
    val req: RequirednessAnalysis = Requiredness(myTable, ctx)
    val rowType = req.lookup(myTable).asInstanceOf[RTable].rowType
    val stage = LowerTableIR.applyTable(myTable, DArrayLowering.All, ctx, req, Map.empty[String, IR])
    val sortFields = sortFieldNames.map(x => SortField(x, Ascending) )
    val sortedTs = LowerDistributedSort.distributedSort(ctx, stage, sortFields, Map.empty[String, IR], rowType)
    val res = TestUtils.eval(sortedTs.mapCollect(Map.empty[String, IR])(x => ToArray(x))).asInstanceOf[IndexedSeq[IndexedSeq[Row]]].flatten

    val rowFunc = myTable.typ.rowType.select(sortFieldNames)._2
    val unsortedCollect = is.hail.expr.ir.TestUtils.collect(myTable)
    val unsortedReq = Requiredness(unsortedCollect, ctx)
    val unsorted = TestUtils.eval(LowerTableIR.apply(unsortedCollect, DArrayLowering.All, ctx, unsortedReq, Map.empty[String, IR])).asInstanceOf[Row](0).asInstanceOf[IndexedSeq[Row]]
    val scalaSorted = unsorted.sortWith{ case (l, r) =>
      val leftKey = rowFunc(l)
      val rightKey = rowFunc(r)
      var ans = false
      var i = 0
      while (i < sortFieldNames.size) {
        if (leftKey(i).asInstanceOf[Int] != rightKey(i).asInstanceOf[Int]) {
          ans = leftKey(i).asInstanceOf[Int] < rightKey(i).asInstanceOf[Int]
          i = sortFieldNames.size
        }
        i += 1
      }
      ans
    }
    assert(res == scalaSorted)
  }

  @Test def testDistributedSort(): Unit = {
    val tableRange = TableRange(100, 10)
    val rangeRow = Ref("row", tableRange.typ.rowType)
    val tableWithExtraField = TableMapRows(
      tableRange,
      MakeStruct(IndexedSeq(
        "idx" -> GetField(rangeRow, "idx"),
        "foo" -> Apply("mod", IndexedSeq(), IndexedSeq(GetField(rangeRow, "idx"), I32(2)), TInt32, ErrorIDs.NO_ERROR),
        "backwards" -> -GetField(rangeRow, "idx"),
        "const" -> I32(4) // Make sure it at least terminates in this scenario.
      ))
    )

    testDistributedSortHelper(tableWithExtraField, IndexedSeq("foo", "idx"))
//    testDistributedSortHelper(tableWithExtraField, IndexedSeq("idx"))
//    testDistributedSortHelper(tableWithExtraField, IndexedSeq("backwards"))
//    testDistributedSortHelper(tableWithExtraField, IndexedSeq("const"))
  }
}
