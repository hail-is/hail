package is.hail.expr.ir.lowering

import is.hail.{ExecStrategy, HailSuite, TestUtils}
import is.hail.expr.ir.{
  mapIR, Apply, Ascending, Descending, ErrorIDs, GetField, I32, Literal, LoweringAnalyses,
  MakeStruct, Ref, SelectFields, SortField, TableIR, TableMapRows, TableRange, ToArray, ToStream,
}
import is.hail.expr.ir.lowering.LowerDistributedSort.samplePartition
import is.hail.types.RTable
import is.hail.types.virtual.{TArray, TInt32, TStruct}
import is.hail.utils.FastSeq

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LowerDistributedSortSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.compileOnly

  @Test def testSamplePartition(): Unit = {
    val dataKeys = IndexedSeq(
      (0, 0),
      (0, -1),
      (1, 4),
      (2, 8),
      (3, 4),
      (4, 5),
      (5, 3),
      (6, 9),
      (7, 7),
      (8, -3),
      (9, 1),
    )
    val elementType = TStruct(("key1", TInt32), ("key2", TInt32), ("value", TInt32))
    val data1 =
      ToStream(Literal(TArray(elementType), dataKeys.map { case (k1, k2) => Row(k1, k2, k1 * k1) }))
    val sampleSeq = ToStream(Literal(TArray(TInt32), IndexedSeq(0, 2, 3, 7)))

    val sampled = samplePartition(
      mapIR(data1)(s => SelectFields(s, IndexedSeq("key1", "key2"))),
      sampleSeq,
      IndexedSeq(SortField("key1", Ascending), SortField("key2", Ascending)),
    )

    assertEvalsTo(
      sampled,
      Row(Row(0, -1), Row(9, 1), IndexedSeq(Row(0, 0), Row(1, 4), Row(2, 8), Row(6, 9)), false),
    )

    val dataKeys2 = IndexedSeq((0, 0), (0, 1), (1, 0), (3, 3))
    val elementType2 = TStruct(("key1", TInt32), ("key2", TInt32))
    val data2 =
      ToStream(Literal(TArray(elementType2), dataKeys2.map { case (k1, k2) => Row(k1, k2) }))
    val sampleSeq2 = ToStream(Literal(TArray(TInt32), IndexedSeq(0)))
    val sampled2 = samplePartition(
      mapIR(data2)(s => SelectFields(s, IndexedSeq("key2", "key1"))),
      sampleSeq2,
      IndexedSeq(SortField("key2", Ascending), SortField("key1", Ascending)),
    )
    assertEvalsTo(sampled2, Row(Row(0, 0), Row(3, 3), IndexedSeq(Row(0, 0)), false))
  }

  // Only does ascending for now
  def testDistributedSortHelper(myTable: TableIR, sortFields: IndexedSeq[SortField]): Unit = {
    val originalShuffleCutoff = backend.getFlag("shuffle_cutoff_to_local_sort")
    try {
      backend.setFlag("shuffle_cutoff_to_local_sort", "40")
      val analyses: LoweringAnalyses = LoweringAnalyses.apply(myTable, ctx)
      val rt = analyses.requirednessAnalysis.lookup(myTable).asInstanceOf[RTable]
      val stage = LowerTableIR.applyTable(myTable, DArrayLowering.All, ctx, analyses)

      val sortedTs = LowerDistributedSort.distributedSort(ctx, stage, sortFields, rt)
        .lower(ctx, myTable.typ.copy(key = FastSeq()))
      val res =
        TestUtils.eval(sortedTs.mapCollect("test")(x => ToArray(x))).asInstanceOf[IndexedSeq[
          IndexedSeq[Row]
        ]].flatten

      val rowFunc = myTable.typ.rowType.select(sortFields.map(_.field))._2
      val unsortedCollect = is.hail.expr.ir.TestUtils.collect(myTable)
      val unsortedAnalyses = LoweringAnalyses.apply(unsortedCollect, ctx)
      val unsorted = TestUtils.eval(LowerTableIR.apply(
        unsortedCollect,
        DArrayLowering.All,
        ctx,
        unsortedAnalyses,
      )).asInstanceOf[Row](0).asInstanceOf[IndexedSeq[Row]]
      val scalaSorted = unsorted.sortWith { case (l, r) =>
        val leftKey = rowFunc(l)
        val rightKey = rowFunc(r)
        var ans = false
        var i = 0
        while (i < sortFields.size) {
          if (leftKey(i).asInstanceOf[Int] != rightKey(i).asInstanceOf[Int]) {
            if (sortFields(i).sortOrder == Ascending) {
              ans = leftKey(i).asInstanceOf[Int] < rightKey(i).asInstanceOf[Int]
            } else {
              ans = leftKey(i).asInstanceOf[Int] > rightKey(i).asInstanceOf[Int]
            }
            i = sortFields.size
          }
          i += 1
        }
        ans
      }
      assert(res == scalaSorted)
    } finally
      backend.setFlag("shuffle_cutoff_to_local_sort", originalShuffleCutoff)
  }

  @Test def testDistributedSort(): Unit = {
    val tableRange = TableRange(100, 10)
    val rangeRow = Ref("row", tableRange.typ.rowType)
    val tableWithExtraField = TableMapRows(
      tableRange,
      MakeStruct(IndexedSeq(
        "idx" -> GetField(rangeRow, "idx"),
        "foo" -> Apply(
          "mod",
          IndexedSeq(),
          IndexedSeq(GetField(rangeRow, "idx"), I32(2)),
          TInt32,
          ErrorIDs.NO_ERROR,
        ),
        "backwards" -> -GetField(rangeRow, "idx"),
        "const" -> I32(4),
      )),
    )

    testDistributedSortHelper(
      tableWithExtraField,
      IndexedSeq(SortField("foo", Ascending), SortField("idx", Ascending)),
    )
    testDistributedSortHelper(tableWithExtraField, IndexedSeq(SortField("idx", Ascending)))
    testDistributedSortHelper(tableWithExtraField, IndexedSeq(SortField("backwards", Ascending)))
    testDistributedSortHelper(tableWithExtraField, IndexedSeq(SortField("const", Ascending)))
    testDistributedSortHelper(tableWithExtraField, IndexedSeq(SortField("idx", Descending)))
    testDistributedSortHelper(
      tableWithExtraField,
      IndexedSeq(SortField("foo", Descending), SortField("idx", Ascending)),
    )
  }

  @Test def testDistributedSortEmpty(): Unit = {
    val tableRange = TableRange(0, 1)
    testDistributedSortHelper(tableRange, IndexedSeq(SortField("idx", Ascending)))
  }
}
