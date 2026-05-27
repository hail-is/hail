package is.hail.expr.ir.lowering

import is.hail.{ExecStrategy, HailSuite, TestUtils}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.expr.ir.{
  makestruct, mapIR, Ascending, Descending, LoweringAnalyses, SortField, TableIR, TableMapRows,
  TableRange,
}
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{
  Apply, ErrorIDs, GetField, I32, Literal, Ref, SelectFields, ToArray, ToStream,
}
import is.hail.expr.ir.lowering.LowerDistributedSort.samplePartition
import is.hail.types.RTable
import is.hail.types.virtual.{TArray, TInt32, TStruct}

import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class LowerDistributedSortSuite extends HailSuite with TestUtils {
  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

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

  @DataProvider(name = "DistributedSort")
  def dataDistributedSort: Array[Array[Any]] = {
    val tableRange = TableRange(100, 10)

    def idx =
      GetField(
        Ref(TableIR.rowName, tableRange.typ.rowType),
        "idx",
      )

    val t =
      TableMapRows(
        tableRange,
        makestruct(
          "idx" -> idx,
          "foo" -> Apply(
            "mod",
            FastSeq(),
            FastSeq(idx, I32(2)),
            TInt32,
            ErrorIDs.NO_ERROR,
          ),
          "ridx" -> -idx,
          "const" -> I32(4),
        ),
      )

    Array(
      Array(TableRange(0, 1), FastSeq(SortField("idx", Ascending))),
      Array(tableRange, FastSeq(SortField("idx", Ascending))),
      Array(t, FastSeq(SortField("idx", Ascending))),
      Array(t, FastSeq(SortField("idx", Descending))),
      Array(t, FastSeq(SortField("ridx", Ascending))),
      Array(t, FastSeq(SortField("const", Ascending))),
      Array(t, FastSeq(SortField("foo", Ascending), SortField("idx", Ascending))),
      Array(t, FastSeq(SortField("foo", Descending), SortField("idx", Ascending))),
    )
  }

  @Test(dataProvider = "DistributedSort")
  def testDistributedSort(tir: TableIR, sortFields: IndexedSeq[SortField]): Unit = {
    ctx.local(flags = ctx.flags + ("shuffle_cutoff_to_local_sort" -> "40")) { ctx =>
      val lowerSorted =
        eval {
          val analyses = LoweringAnalyses(tir, ctx)
          val rt = analyses.requirednessAnalysis.lookup(tir).asInstanceOf[RTable]
          val stage = LowerTableIR.applyTable(tir, DArrayLowering.All, ctx, analyses)

          LowerDistributedSort
            .distributedSort(ctx, stage, sortFields, rt)
            .lower(ctx, tir.typ.copy(key = FastSeq()))
            .mapCollect("test")(ToArray(_))
        }
          .asInstanceOf[IndexedSeq[IndexedSeq[Row]]]
          .flatten

      val unsorted =
        eval {
          val ir = collect(tir)
          val analyses = LoweringAnalyses(ir, ctx)
          LowerTableIR.apply(ir, DArrayLowering.All, ctx, analyses)
        }
          .asInstanceOf[Row](0)
          .asInstanceOf[IndexedSeq[Row]]

      val rowFunc =
        tir.typ.rowType.select(sortFields.map(_.field))._2

      val scalaSorted =
        unsorted.sortWith { (l, r) =>
          val leftKey = rowFunc(l)
          val rightKey = rowFunc(r)

          sortFields.indices.collectFirst {
            case i if leftKey(i) != rightKey(i) =>
              val L = leftKey(i).asInstanceOf[Int]
              val R = rightKey(i).asInstanceOf[Int]
              if (sortFields(i).sortOrder == Ascending) L < R else L > R
          }.getOrElse(false)
        }

      assert(lowerSorted == scalaSorted)
    }
  }
}
