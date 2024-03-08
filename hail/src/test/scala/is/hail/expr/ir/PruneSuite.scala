package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.expr.Nat
import is.hail.methods.{ForceCountMatrixTable, ForceCountTable}
import is.hail.rvd.RVD
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.Row
import org.json4s.JValue
import org.testng.annotations.{DataProvider, Test}

class PruneSuite extends HailSuite {
  @Test def testUnionType(): Unit = {
    val base = TStruct(
      "a" -> TStruct(
        "aa" -> TInt32,
        "ab" -> TStruct(
          "aaa" -> TString
        ),
      ),
      "b" -> TInt32,
      "c" -> TArray(TStruct(
        "ca" -> TInt32
      )),
    )

    assert(PruneDeadFields.unify(base, TStruct.empty) == TStruct.empty)
    assert(PruneDeadFields.unify(base, TStruct("b" -> TInt32)) == TStruct("b" -> TInt32))
    assert(
      PruneDeadFields.unify(base, TStruct("a" -> TStruct.empty)) == TStruct("a" -> TStruct.empty)
    )
    assert(PruneDeadFields.unify(
      base,
      TStruct("a" -> TStruct.empty),
      TStruct("b" -> TInt32),
    ) == TStruct("a" -> TStruct.empty, "b" -> TInt32))
    assert(PruneDeadFields.unify(base, TStruct("c" -> TArray(TStruct.empty))) == TStruct(
      "c" -> TArray(TStruct.empty)
    ))
    assert(PruneDeadFields.unify(
      base,
      TStruct("a" -> TStruct("ab" -> TStruct.empty)),
      TStruct("c" -> TArray(TStruct.empty)),
    ) == TStruct("a" -> TStruct("ab" -> TStruct.empty), "c" -> TArray(TStruct.empty)))
  }

  @Test def testIsSupertype(): Unit = {
    val emptyTuple = TTuple.empty
    val tuple1Int = TTuple(TInt32)
    val tuple2Ints = TTuple(TInt32, TInt32)
    val tuple2IntsFirstRemoved = TTuple(IndexedSeq(TupleField(1, TInt32)))

    assert(PruneDeadFields.isSupertype(emptyTuple, tuple2Ints))
    assert(PruneDeadFields.isSupertype(tuple1Int, tuple2Ints))
    assert(PruneDeadFields.isSupertype(tuple2IntsFirstRemoved, tuple2Ints))
  }

  @Test def testIsSupertypeWithDistinctFieldTypes(): Unit = {
    val tuple2Ints = TTuple(TInt32, TFloat64)
    val tuple2IntsFirstRemoved = TTuple(IndexedSeq(TupleField(1, TFloat64)))

    assert(PruneDeadFields.isSupertype(tuple2IntsFirstRemoved, tuple2Ints))
  }

  def checkMemo(ir: BaseIR, requestedType: BaseType, expected: Array[BaseType]): Unit = {
    val irCopy = ir.deepCopy()
    assert(
      PruneDeadFields.isSupertype(requestedType, irCopy.typ),
      s"not supertype:\n  super: ${requestedType.parsableString()}\n  sub:   ${irCopy.typ.parsableString()}",
    )
    val ms = PruneDeadFields.ComputeMutableState(Memo.empty[BaseType], mutable.HashMap.empty)
    irCopy match {
      case mir: MatrixIR =>
        PruneDeadFields.memoizeMatrixIR(ctx, mir, requestedType.asInstanceOf[MatrixType], ms)
      case tir: TableIR =>
        PruneDeadFields.memoizeTableIR(ctx, tir, requestedType.asInstanceOf[TableType], ms)
      case ir: IR => PruneDeadFields.memoizeValueIR(ctx, ir, requestedType.asInstanceOf[Type], ms)
    }
    irCopy.children.zipWithIndex.foreach { case (child, i) =>
      if (expected(i) != null && expected(i) != ms.requestedType.lookup(child)) {
        fatal(
          s"For base IR $ir\n  Child $i with IR $child\n  Expected: ${expected(i)}\n  Actual:   ${ms.requestedType.lookup(child)}"
        )
      }
    }
  }

  def checkRebuild[T <: BaseIR](
    ir: T,
    requestedType: BaseType,
    f: (T, T) => Boolean = (left: T, right: T) => left == right,
  ): Unit = {
    val irCopy = ir.deepCopy()
    val ms = PruneDeadFields.ComputeMutableState(Memo.empty[BaseType], mutable.HashMap.empty)
    val rebuilt = (irCopy match {
      case mir: MatrixIR =>
        PruneDeadFields.memoizeMatrixIR(ctx, mir, requestedType.asInstanceOf[MatrixType], ms)
        PruneDeadFields.rebuild(ctx, mir, ms.rebuildState)
      case tir: TableIR =>
        PruneDeadFields.memoizeTableIR(ctx, tir, requestedType.asInstanceOf[TableType], ms)
        PruneDeadFields.rebuild(ctx, tir, ms.rebuildState)
      case ir: IR =>
        PruneDeadFields.memoizeValueIR(ctx, ir, requestedType.asInstanceOf[Type], ms)
        PruneDeadFields.rebuildIR(
          ctx,
          ir,
          BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)),
          ms.rebuildState,
        )
    }).asInstanceOf[T]
    if (!f(ir, rebuilt))
      fatal(s"IR did not rebuild the same:\n  Base:    $ir\n  Rebuilt: $rebuilt")
  }

  lazy val tab = TableLiteral(
    TableKeyBy(
      TableParallelize(
        Literal(
          TStruct(
            "rows" -> TArray(TStruct(
              "1" -> TString,
              "2" -> TArray(TStruct("2A" -> TInt32)),
              "3" -> TString,
              "4" -> TStruct("A" -> TInt32, "B" -> TArray(TStruct("i" -> TString))),
              "5" -> TString,
            )),
            "global" -> TStruct("g1" -> TInt32, "g2" -> TInt32),
          ),
          Row(
            FastSeq(Row("hi", FastSeq(Row(1)), "bye", Row(2, FastSeq(Row("bar"))), "foo")),
            Row(5, 10),
          ),
        ),
        None,
      ),
      FastSeq("3"),
      false,
    ).analyzeAndExecute(ctx).asTableValue(ctx),
    theHailClassLoader,
  )

  lazy val tr = TableRead(
    tab.typ,
    false,
    new FakeTableReader {
      override def pathsUsed: Seq[String] = Seq.empty
      override def fullType: TableType = tab.typ
    },
  )

  lazy val mType = MatrixType(
    TStruct("g1" -> TInt32, "g2" -> TFloat64),
    FastSeq("ck"),
    TStruct("ck" -> TString, "c2" -> TInt32, "c3" -> TArray(TStruct("cc" -> TInt32))),
    FastSeq("rk"),
    TStruct(
      "rk" -> TInt32,
      "r2" -> TStruct("x" -> TInt32),
      "r3" -> TArray(TStruct("rr" -> TInt32)),
    ),
    TStruct("e1" -> TFloat64, "e2" -> TFloat64),
  )

  lazy val mat = MatrixLiteral(
    ctx,
    mType,
    RVD.empty(ctx, mType.canonicalTableType.canonicalRVDType),
    Row(1, 1.0),
    FastSeq(Row("1", 2, FastSeq(Row(3)))),
  )

  lazy val mr = MatrixRead(
    mat.typ,
    false,
    false,
    new MatrixReader {
      def pathsUsed: IndexedSeq[String] = FastSeq()

      override def columnCount: Option[Int] = None

      def partitionCounts: Option[IndexedSeq[Long]] = None

      def rowUIDType = TTuple(TInt64, TInt64)
      def colUIDType = TTuple(TInt64, TInt64)

      def fullMatrixTypeWithoutUIDs: MatrixType = mat.typ

      def lower(requestedType: MatrixType, dropCols: Boolean, dropRows: Boolean): TableIR = ???

      def toJValue: JValue = ???

      override def renderShort(): String = ???
    },
  )

  lazy val emptyTableDep = TableType(TStruct.empty, FastSeq(), TStruct.empty)

  def tableRefBoolean(tt: TableType, fields: String*): IR = {
    var let: IR = True()
    fields.foreach { f =>
      val split = f.split("\\.")
      var ir: IR = split(0) match {
        case "row" => Ref("row", tt.rowType)
        case "global" => Ref("global", tt.globalType)
      }

      split.tail.foreach(field => ir = GetField(ir, field))
      let = Let(FastSeq(genUID() -> ir), let)
    }
    let
  }

  def tableRefStruct(tt: TableType, fields: String*): IR =
    MakeStruct(tt.key.map(k => k -> GetField(Ref("row", tt.rowType), k)) ++ FastSeq(
      "foo" -> tableRefBoolean(tt, fields: _*)
    ))

  def matrixRefBoolean(mt: MatrixType, fields: String*): IR = {
    var let: IR = True()
    fields.foreach { f =>
      val split = f.split("\\.")
      var ir: IR = split(0) match {
        case "va" => Ref("va", mt.rowType)
        case "sa" => Ref("sa", mt.colType)
        case "g" => Ref("g", mt.entryType)
        case "global" => Ref("global", mt.globalType)
      }

      split.tail.foreach(field => ir = GetField(ir, field))
      let = Let(FastSeq(genUID() -> ir), let)
    }
    let
  }

  def matrixRefStruct(mt: MatrixType, fields: String*): IR =
    MakeStruct(FastSeq("foo" -> matrixRefBoolean(mt, fields: _*)))

  def subsetTable(tt: TableType, fields: String*): TableType = {
    val rowFields = new BoxedArrayBuilder[TStruct]()
    val globalFields = new BoxedArrayBuilder[TStruct]()
    var noKey = false
    fields.foreach { f =>
      val split = f.split("\\.")
      split(0) match {
        case "row" =>
          rowFields += PruneDeadFields.subsetType(tt.rowType, split, 1).asInstanceOf[TStruct]
        case "global" =>
          globalFields += PruneDeadFields.subsetType(tt.globalType, split, 1).asInstanceOf[TStruct]
        case "NO_KEY" =>
          noKey = true
      }
    }
    val k = if (noKey) FastSeq() else tt.key
    tt.copy(
      key = k,
      rowType = PruneDeadFields.unify(
        tt.rowType,
        Array(PruneDeadFields.selectKey(tt.rowType, k)) ++ rowFields.result(): _*
      ),
      globalType = PruneDeadFields.unify(tt.globalType, globalFields.result(): _*),
    )
  }

  def subsetMatrixTable(mt: MatrixType, fields: String*): MatrixType = {
    val rowFields = new BoxedArrayBuilder[TStruct]()
    val colFields = new BoxedArrayBuilder[TStruct]()
    val entryFields = new BoxedArrayBuilder[TStruct]()
    val globalFields = new BoxedArrayBuilder[TStruct]()
    var noRowKey = false
    var noColKey = false
    fields.foreach { f =>
      val split = f.split("\\.")
      split(0) match {
        case "va" =>
          rowFields += PruneDeadFields.subsetType(mt.rowType, split, 1).asInstanceOf[TStruct]
        case "sa" =>
          colFields += PruneDeadFields.subsetType(mt.colType, split, 1).asInstanceOf[TStruct]
        case "g" =>
          entryFields += PruneDeadFields.subsetType(mt.entryType, split, 1).asInstanceOf[TStruct]
        case "global" =>
          globalFields += PruneDeadFields.subsetType(mt.globalType, split, 1).asInstanceOf[TStruct]
        case "NO_ROW_KEY" =>
          noRowKey = true
        case "NO_COL_KEY" =>
          noColKey = true
      }
    }
    val ck = if (noColKey) FastSeq() else mt.colKey
    val rk = if (noRowKey) FastSeq() else mt.rowKey
    MatrixType(
      rowKey = rk,
      colKey = ck,
      globalType = PruneDeadFields.unify(mt.globalType, globalFields.result(): _*),
      colType = PruneDeadFields.unify(
        mt.colType,
        Array(PruneDeadFields.selectKey(mt.colType, ck)) ++ colFields.result(): _*
      ),
      rowType = PruneDeadFields.unify(
        mt.rowType,
        Array(PruneDeadFields.selectKey(mt.rowType, rk)) ++ rowFields.result(): _*
      ),
      entryType = PruneDeadFields.unify(mt.entryType, entryFields.result(): _*),
    )
  }

  def mangle(t: TableIR): TableIR =
    TableRename(
      t,
      t.typ.rowType.fieldNames.map(x => x -> (x + "_")).toMap,
      t.typ.globalType.fieldNames.map(x => x -> (x + "_")).toMap,
    )

  @Test def testTableJoinMemo(): Unit = {
    val tk1 = TableKeyBy(tab, Array("1"))
    val tk2 = mangle(TableKeyBy(tab, Array("3")))
    val tj = TableJoin(tk1, tk2, "inner", 1)
    checkMemo(
      tj,
      subsetTable(tj.typ, "row.1", "row.4", "row.1_"),
      Array(
        subsetTable(tk1.typ, "row.1", "row.4"),
        subsetTable(tk2.typ, "row.1_", "row.3_"),
      ),
    )

    val tk3 = TableKeyBy(tab, Array("1", "2"))
    val tk4 = mangle(TableKeyBy(tab, Array("1", "2")))

    val tj2 = TableJoin(tk3, tk4, "inner", 1)
    checkMemo(
      tj2,
      subsetTable(tj2.typ, "row.3_"),
      Array(
        subsetTable(tk3.typ, "row.1", "row.2"),
        subsetTable(tk4.typ, "row.1_", "row.2_", "row.3_"),
      ),
    )

    checkMemo(
      tj2,
      subsetTable(tj2.typ, "row.3_", "NO_KEY"),
      Array(
        TableType(globalType = TStruct.empty, key = Array("1"), rowType = TStruct("1" -> TString)),
        TableType(
          globalType = TStruct.empty,
          key = Array("1_"),
          rowType = TStruct("1_" -> TString, "3_" -> TString),
        ),
      ),
    )
  }

  @Test def testTableLeftJoinRightDistinctMemo(): Unit = {
    val tk1 = TableKeyBy(tab, Array("1"))
    val tk2 = TableKeyBy(tab, Array("3"))
    val tj = TableLeftJoinRightDistinct(tk1, tk2, "foo")
    checkMemo(
      tj,
      subsetTable(tj.typ, "row.1", "row.4", "row.foo"),
      Array(
        subsetTable(tk1.typ, "row.1", "row.4"),
        subsetTable(tk2.typ),
      ),
    )
  }

  @Test def testTableIntervalJoinMemo(): Unit = {
    val tk1 = TableKeyBy(tab, Array("1"))
    val tk2 = TableKeyBy(tab, Array("3"))
    val tj = TableIntervalJoin(tk1, tk2, "foo", product = false)
    checkMemo(
      tj,
      subsetTable(tj.typ, "row.1", "row.4", "row.foo"),
      Array(
        subsetTable(tk1.typ, "row.1", "row.4"),
        subsetTable(tk2.typ),
      ),
    )
  }

  @Test def testTableMultiWayZipJoinMemo(): Unit = {
    val tk1 = TableKeyBy(tab, Array("1"))
    val ts = Array(tk1, tk1, tk1)
    val tmwzj = TableMultiWayZipJoin(ts, "data", "gbls")
    checkMemo(
      tmwzj,
      subsetTable(tmwzj.typ, "row.data.2", "global.gbls.g1"),
      ts.map(t => subsetTable(t.typ, "row.2", "global.g1")),
    )
  }

  @Test def testTableExplodeMemo(): Unit = {
    val te = TableExplode(tab, Array("2"))
    checkMemo(te, subsetTable(te.typ), Array(subsetTable(tab.typ, "row.2")))
  }

  @Test def testTableFilterMemo(): Unit = {
    checkMemo(
      TableFilter(tab, tableRefBoolean(tab.typ, "row.2")),
      subsetTable(tab.typ, "row.3"),
      Array(subsetTable(tab.typ, "row.2", "row.3"), null),
    )
    checkMemo(
      TableFilter(tab, False()),
      subsetTable(tab.typ, "row.1"),
      Array(subsetTable(tab.typ, "row.1"), TBoolean),
    )
  }

  @Test def testTableKeyByMemo(): Unit = {
    val tk = TableKeyBy(tab, Array("1"))
    checkMemo(
      tk,
      subsetTable(tk.typ, "row.2"),
      Array(subsetTable(tab.typ, "row.1", "row.2", "NO_KEY")),
    )

    val tk2 = TableKeyBy(tab, Array("3"), isSorted = true)
    checkMemo(tk2, subsetTable(tk2.typ, "row.2"), Array(subsetTable(tab.typ, "row.2")))

  }

  @Test def testTableMapRowsMemo(): Unit = {
    val tmr = TableMapRows(tab, tableRefStruct(tab.typ, "row.1", "row.2"))
    checkMemo(
      tmr,
      subsetTable(tmr.typ, "row.foo"),
      Array(subsetTable(tab.typ, "row.1", "row.2"), null),
    )

    val tmr2 = TableMapRows(tab, tableRefStruct(tab.typ, "row.1", "row.2"))
    checkMemo(
      tmr2,
      subsetTable(tmr2.typ, "row.foo", "NO_KEY"),
      Array(subsetTable(tab.typ, "row.1", "row.2", "NO_KEY"), null),
    )
  }

  @Test def testTableMapGlobalsMemo(): Unit = {
    val tmg = TableMapGlobals(tab, tableRefStruct(tab.typ, "global.g1"))
    checkMemo(
      tmg,
      subsetTable(tmg.typ, "global.foo"),
      Array(subsetTable(tab.typ, "global.g1"), null),
    )
  }

  @Test def testMatrixColsTableMemo(): Unit = {
    val mct = MatrixColsTable(mat)
    checkMemo(
      mct,
      subsetTable(mct.typ, "global.g1", "row.c2"),
      Array(subsetMatrixTable(mat.typ, "global.g1", "sa.c2", "NO_ROW_KEY")),
    )
  }

  @Test def testMatrixRowsTableMemo(): Unit = {
    val mrt = MatrixRowsTable(mat)
    checkMemo(
      mrt,
      subsetTable(mrt.typ, "global.g1", "row.r2"),
      Array(subsetMatrixTable(mat.typ, "global.g1", "va.r2", "NO_COL_KEY")),
    )
  }

  @Test def testMatrixEntriesTableMemo(): Unit = {
    val met = MatrixEntriesTable(mat)
    checkMemo(
      met,
      subsetTable(met.typ, "global.g1", "row.r2", "row.c2", "row.e2"),
      Array(subsetMatrixTable(mat.typ, "global.g1", "va.r2", "sa.c2", "g.e2")),
    )
  }

  @Test def testTableKeyByAndAggregateMemo(): Unit = {
    val tka = TableKeyByAndAggregate(
      tab,
      tableRefStruct(tab.typ, "row.2"),
      MakeStruct(FastSeq("bar" -> tableRefBoolean(tab.typ, "row.3"))),
      None,
      1,
    )

    checkMemo(
      tka,
      subsetTable(tka.typ, "row.foo"),
      Array(subsetTable(tab.typ, "row.2", "row.3", "NO_KEY"), null, null),
    )
    checkMemo(tka, subsetTable(tka.typ), Array(subsetTable(tab.typ, "row.3", "NO_KEY"), null, null))
  }

  @Test def testTableAggregateByKeyMemo(): Unit = {
    val tabk = TableAggregateByKey(
      tab,
      SelectFields(Ref("row", tab.typ.rowType), IndexedSeq("5")),
    )
    checkMemo(
      tabk,
      requestedType = subsetTable(tabk.typ, "row.3", "row.5"),
      Array(subsetTable(tabk.typ, "row.3", "row.5"), TStruct(("5", TString))),
    )
  }

  @Test def testTableUnionMemo(): Unit =
    checkMemo(
      TableUnion(FastSeq(tab, tab)),
      subsetTable(tab.typ, "row.1", "global.g1"),
      Array(subsetTable(tab.typ, "row.1", "global.g1"), subsetTable(tab.typ, "row.1")),
    )

  @Test def testTableOrderByMemo(): Unit = {
    val tob = TableOrderBy(tab, Array(SortField("2", Ascending)))
    checkMemo(tob, subsetTable(tob.typ), Array(subsetTable(tab.typ, "row.2", "row.2.2A", "NO_KEY")))

    val tob2 = TableOrderBy(tab, Array(SortField("3", Ascending)))
    checkMemo(tob2, subsetTable(tob2.typ), Array(subsetTable(tab.typ)))
  }

  @Test def testCastMatrixToTableMemo(): Unit = {
    val m2t = CastMatrixToTable(mat, "__entries", "__cols")
    checkMemo(
      m2t,
      subsetTable(m2t.typ, "row.r2", "global.__cols.c2", "global.g2", "row.__entries.e2"),
      Array(subsetMatrixTable(mat.typ, "va.r2", "global.g2", "sa.c2", "g.e2", "NO_COL_KEY")),
    )
  }

  @Test def testMatrixFilterColsMemo(): Unit = {
    val mfc = MatrixFilterCols(mat, matrixRefBoolean(mat.typ, "global.g1", "sa.c2"))
    checkMemo(
      mfc,
      subsetMatrixTable(mfc.typ, "sa.c3"),
      Array(subsetMatrixTable(mat.typ, "global.g1", "sa.c2", "sa.c3"), null),
    )
  }

  @Test def testMatrixFilterRowsMemo(): Unit = {
    val mfr = MatrixFilterRows(mat, matrixRefBoolean(mat.typ, "global.g1", "va.r2"))
    checkMemo(
      mfr,
      subsetMatrixTable(mfr.typ, "sa.c3", "va.r3"),
      Array(subsetMatrixTable(mat.typ, "global.g1", "va.r2", "sa.c3", "va.r3"), null),
    )
  }

  @Test def testMatrixFilterEntriesMemo(): Unit = {
    val mfe =
      MatrixFilterEntries(mat, matrixRefBoolean(mat.typ, "global.g1", "va.r2", "sa.c2", "g.e2"))
    checkMemo(
      mfe,
      subsetMatrixTable(mfe.typ, "sa.c3", "va.r3"),
      Array(
        subsetMatrixTable(mat.typ, "global.g1", "va.r2", "sa.c3", "sa.c2", "va.r3", "g.e2"),
        null,
      ),
    )
  }

  @Test def testMatrixMapColsMemo(): Unit = {
    val mmc = MatrixMapCols(
      mat,
      matrixRefStruct(mat.typ, "global.g1", "sa.c2", "va.r2", "g.e2"),
      Some(FastSeq()),
    )
    checkMemo(
      mmc,
      subsetMatrixTable(mmc.typ, "va.r3", "sa.foo"),
      Array(
        subsetMatrixTable(mat.typ, "global.g1", "sa.c2", "va.r2", "g.e2", "va.r3", "NO_COL_KEY"),
        null,
      ),
    )
    val mmc2 = MatrixMapCols(
      mat,
      MakeStruct(FastSeq(
        ("ck" -> GetField(Ref("sa", mat.typ.colType), "ck")),
        ("foo", matrixRefStruct(mat.typ, "global.g1", "sa.c2", "va.r2", "g.e2")),
      )),
      None,
    )
    checkMemo(
      mmc2,
      subsetMatrixTable(mmc2.typ, "va.r3", "sa.foo.foo"),
      Array(subsetMatrixTable(mat.typ, "global.g1", "sa.c2", "va.r2", "g.e2", "va.r3"), null),
    )
  }

  @Test def testMatrixKeyRowsByMemo(): Unit = {
    val mkr = MatrixKeyRowsBy(mat, FastSeq("rk"))
    checkMemo(mkr, subsetMatrixTable(mkr.typ, "va.rk"), Array(subsetMatrixTable(mat.typ, "va.rk")))
  }

  @Test def testMatrixMapRowsMemo(): Unit = {
    val mmr = MatrixMapRows(
      MatrixKeyRowsBy(mat, IndexedSeq.empty),
      matrixRefStruct(mat.typ, "global.g1", "sa.c2", "va.r2", "g.e2"),
    )
    checkMemo(
      mmr,
      subsetMatrixTable(mmr.typ, "sa.c3", "va.foo"),
      Array(
        subsetMatrixTable(
          mat.typ.copy(rowKey = IndexedSeq.empty),
          "global.g1",
          "sa.c2",
          "va.r2",
          "g.e2",
          "sa.c3",
        ),
        null,
      ),
    )
  }

  @Test def testMatrixMapGlobalsMemo(): Unit = {
    val mmg = MatrixMapGlobals(mat, matrixRefStruct(mat.typ, "global.g1"))
    checkMemo(
      mmg,
      subsetMatrixTable(mmg.typ, "global.foo", "va.r3", "sa.c3"),
      Array(subsetMatrixTable(mat.typ, "global.g1", "va.r3", "sa.c3"), null),
    )
  }

  @Test def testMatrixAnnotateRowsTableMemo(): Unit = {
    val tl = TableLiteral(Interpret(MatrixRowsTable(mat), ctx), theHailClassLoader)
    val mart = MatrixAnnotateRowsTable(mat, tl, "foo", product = false)
    checkMemo(
      mart,
      subsetMatrixTable(mart.typ, "va.foo.r3", "va.r3"),
      Array(subsetMatrixTable(mat.typ, "va.r3"), subsetTable(tl.typ, "row.r3")),
    )
  }

  @Test def testCollectColsByKeyMemo(): Unit = {
    val ccbk = MatrixCollectColsByKey(mat)
    checkMemo(
      ccbk,
      subsetMatrixTable(ccbk.typ, "g.e2", "sa.c2", "NO_COL_KEY"),
      Array(subsetMatrixTable(mat.typ, "g.e2", "sa.c2")),
    )
  }

  @Test def testMatrixExplodeRowsMemo(): Unit = {
    val mer = MatrixExplodeRows(mat, FastSeq("r3"))
    checkMemo(
      mer,
      subsetMatrixTable(mer.typ, "va.r2"),
      Array(subsetMatrixTable(mat.typ, "va.r2", "va.r3")),
    )
  }

  @Test def testMatrixRepartitionMemo(): Unit = {
    checkMemo(
      MatrixRepartition(mat, 10, RepartitionStrategy.SHUFFLE),
      subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
      Array(
        subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
        subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
      ),
    )
  }

  @Test def testMatrixUnionRowsMemo(): Unit = {
    checkMemo(
      MatrixUnionRows(FastSeq(mat, mat)),
      subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
      Array(
        subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
        subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
      ),
    )
  }

  @Test def testMatrixDistinctByRowMemo(): Unit = {
    checkMemo(
      MatrixDistinctByRow(mat),
      subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
      Array(
        subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
        subsetMatrixTable(mat.typ, "va.r2", "global.g1"),
      ),
    )
  }

  @Test def testMatrixExplodeColsMemo(): Unit = {
    val mer = MatrixExplodeCols(mat, FastSeq("c3"))
    checkMemo(
      mer,
      subsetMatrixTable(mer.typ, "va.r2"),
      Array(subsetMatrixTable(mat.typ, "va.r2", "sa.c3")),
    )
  }

  @Test def testCastTableToMatrixMemo(): Unit = {
    val m2t = CastMatrixToTable(mat, "__entries", "__cols")
    val t2m = CastTableToMatrix(m2t, "__entries", "__cols", FastSeq("ck"))
    checkMemo(
      t2m,
      subsetMatrixTable(mat.typ, "va.r2", "sa.c2", "global.g2", "g.e2"),
      Array(subsetTable(
        m2t.typ,
        "row.r2",
        "global.g2",
        "global.__cols.ck",
        "global.__cols.c2",
        "row.__entries.e2",
      )),
    )
  }

  @Test def testMatrixAggregateRowsByKeyMemo(): Unit = {
    val magg = MatrixAggregateRowsByKey(
      mat,
      matrixRefStruct(mat.typ, "g.e2", "va.r2", "sa.c2"),
      matrixRefStruct(mat.typ, "va.r3", "global.g1"),
    )
    checkMemo(
      magg,
      subsetMatrixTable(magg.typ, "sa.c3", "g.foo", "va.foo"),
      Array(
        subsetMatrixTable(mat.typ, "sa.c3", "g.e2", "va.r2", "sa.c2", "global.g1", "va.r3"),
        null,
        null,
      ),
    )
  }

  @Test def testMatrixAggregateColsByKeyMemo(): Unit = {
    val magg = MatrixAggregateColsByKey(
      mat,
      matrixRefStruct(mat.typ, "g.e2", "va.r2", "sa.c2"),
      matrixRefStruct(mat.typ, "sa.c3", "global.g1"),
    )
    checkMemo(
      magg,
      subsetMatrixTable(magg.typ, "va.r3", "g.foo", "sa.foo"),
      Array(
        subsetMatrixTable(mat.typ, "sa.c2", "va.r2", "va.r3", "g.e2", "global.g1", "sa.c3"),
        null,
        null,
      ),
    )
  }

  val ref = Ref("x", TStruct("a" -> TInt32, "b" -> TInt32, "c" -> TInt32))
  val arr = MakeArray(FastSeq(ref, ref), TArray(ref.typ))
  val st = MakeStream(FastSeq(ref, ref), TStream(ref.typ))
  val ndArr = MakeNDArray(arr, MakeTuple(IndexedSeq((0, I64(2L)))), True(), ErrorIDs.NO_ERROR)
  val empty = TStruct.empty
  val justA = TStruct("a" -> TInt32)
  val justB = TStruct("b" -> TInt32)
  val aAndB = TStruct("a" -> TInt32, "b" -> TInt32)
  val bAndA = TStruct("b" -> TInt32, "a" -> TInt32)
  val justARequired = TStruct("a" -> TInt32)
  val justBRequired = TStruct("b" -> TInt32)

  @Test def testIfMemo(): Unit =
    checkMemo(If(True(), ref, ref), justA, Array(TBoolean, justA, justA))

  @Test def testSwitchMemo(): Unit =
    checkMemo(
      Switch(I32(0), ref, FastSeq(ref)),
      justA,
      Array(TInt32, justA, justA),
    )

  @Test def testCoalesceMemo(): Unit =
    checkMemo(Coalesce(FastSeq(ref, ref)), justA, Array(justA, justA))

  @Test def testLetMemo(): Unit = {
    checkMemo(Let(FastSeq("foo" -> ref), Ref("foo", ref.typ)), justA, Array(justA, null))
    checkMemo(Let(FastSeq("foo" -> ref), True()), TBoolean, Array(empty, null))
  }

  @Test def testAggLetMemo(): Unit = {
    checkMemo(
      AggLet(
        "foo",
        ref,
        ApplyAggOp(
          FastSeq(),
          FastSeq(
            SelectFields(Ref("foo", ref.typ), IndexedSeq("a"))
          ),
          AggSignature(Collect(), FastSeq(), FastSeq(ref.typ)),
        ),
        false,
      ),
      TArray(justA),
      Array(justA, null),
    )
    checkMemo(AggLet("foo", ref, True(), false), TBoolean, Array(empty, null))
  }

  @Test def testMakeArrayMemo(): Unit =
    checkMemo(arr, TArray(justB), Array(justB, justB))

  @Test def testArrayRefMemo(): Unit =
    checkMemo(ArrayRef(arr, I32(0)), justB, Array(TArray(justB), null, null))

  @Test def testArrayLenMemo(): Unit =
    checkMemo(ArrayLen(arr), TInt32, Array(TArray(empty)))

  @Test def testStreamTakeMemo(): Unit =
    checkMemo(StreamTake(st, I32(2)), TStream(justA), Array(TStream(justA), null))

  @Test def testStreamDropMemo(): Unit =
    checkMemo(StreamDrop(st, I32(2)), TStream(justA), Array(TStream(justA), null))

  @Test def testStreamMapMemo(): Unit =
    checkMemo(
      StreamMap(st, "foo", Ref("foo", ref.typ)),
      TStream(justB),
      Array(TStream(justB), null),
    )

  @Test def testStreamGroupedMemo(): Unit =
    checkMemo(StreamGrouped(st, I32(2)), TStream(TStream(justB)), Array(TStream(justB), null))

  @Test def testStreamGroupByKeyMemo(): Unit =
    checkMemo(
      StreamGroupByKey(st, FastSeq("a"), false),
      TStream(TStream(justB)),
      Array(TStream(TStruct("a" -> TInt32, "b" -> TInt32)), null),
    )

  @Test def testStreamMergeMemo(): Unit = {
    val st2 = st.deepCopy()
    checkMemo(
      StreamMultiMerge(
        IndexedSeq(st, st2),
        FastSeq("a"),
      ),
      TStream(justB),
      Array(TStream(aAndB), TStream(aAndB)),
    )
  }

  @Test def testStreamZipMemo(): Unit = {
    val a2 = st.deepCopy()
    val a3 = st.deepCopy()
    for (
      b <- Array(
        ArrayZipBehavior.ExtendNA,
        ArrayZipBehavior.TakeMinLength,
        ArrayZipBehavior.AssertSameLength,
      )
    ) {
      checkMemo(
        StreamZip(
          FastSeq(st, a2, a3),
          FastSeq("foo", "bar", "baz"),
          Let(
            FastSeq(
              "foo1" -> GetField(Ref("foo", ref.typ), "b"),
              "bar2" -> GetField(Ref("bar", ref.typ), "a"),
            ),
            False(),
          ),
          b,
        ),
        TStream(TBoolean),
        Array(TStream(justB), TStream(justA), TStream(empty), null),
      )
    }

    checkMemo(
      StreamZip(
        FastSeq(st, a2, a3),
        FastSeq("foo", "bar", "baz"),
        Let(
          FastSeq(
            "foo1" -> GetField(Ref("foo", ref.typ), "b"),
            "bar2" -> GetField(Ref("bar", ref.typ), "a"),
          ),
          False(),
        ),
        ArrayZipBehavior.AssumeSameLength,
      ),
      TStream(TBoolean),
      Array(TStream(justB), TStream(justA), null, null),
    )
  }

  @Test def testStreamFilterMemo(): Unit = {
    checkMemo(
      StreamFilter(st, "foo", Let(FastSeq("foo2" -> GetField(Ref("foo", ref.typ), "b")), False())),
      TStream(empty),
      Array(TStream(justB), null),
    )
    checkMemo(StreamFilter(st, "foo", False()), TStream(empty), Array(TStream(empty), null))
    checkMemo(StreamFilter(st, "foo", False()), TStream(justB), Array(TStream(justB), null))
  }

  @Test def testStreamFlatMapMemo(): Unit =
    checkMemo(
      StreamFlatMap(st, "foo", MakeStream(FastSeq(Ref("foo", ref.typ)), TStream(ref.typ))),
      TStream(justA),
      Array(TStream(justA), null),
    )

  @Test def testStreamFoldMemo(): Unit =
    checkMemo(
      StreamFold(st, I32(0), "comb", "foo", GetField(Ref("foo", ref.typ), "a")),
      TInt32,
      Array(TStream(justA), null, null),
    )

  @Test def testStreamScanMemo(): Unit =
    checkMemo(
      StreamScan(st, I32(0), "comb", "foo", GetField(Ref("foo", ref.typ), "a")),
      TStream(TInt32),
      Array(TStream(justA), null, null),
    )

  @Test def testStreamJoinRightDistinct(): Unit = {
    val l = Ref("l", ref.typ)
    val r = Ref("r", ref.typ)
    checkMemo(
      StreamJoinRightDistinct(
        st,
        st,
        FastSeq("a"),
        FastSeq("a"),
        "l",
        "r",
        MakeStruct(FastSeq(
          "a" -> GetField(l, "a"),
          "b" -> GetField(l, "b"),
          "c" -> GetField(l, "c"),
          "d" -> GetField(r, "b"),
          "e" -> GetField(r, "c"),
        )),
        "left",
      ),
      TStream(TStruct("b" -> TInt32, "d" -> TInt32)),
      Array(
        TStream(TStruct("a" -> TInt32, "b" -> TInt32)),
        TStream(TStruct("a" -> TInt32, "b" -> TInt32)),
        TStruct("b" -> TInt32, "d" -> TInt32),
      ),
    )
  }

  @Test def testStreamLeftIntervalJoin(): Unit = {
    val leftElemType = TStruct("a" -> TInt32, "b" -> TInt32, "c" -> TInt32)
    val rightElemType = TStruct("interval" -> TInterval(TInt32), "ignored" -> TVoid)

    val join =
      StreamLeftIntervalJoin(
        MakeStream(FastSeq(), TStream(leftElemType)),
        MakeStream(FastSeq(), TStream(rightElemType)),
        leftElemType.fieldNames.head,
        "interval",
        "lname",
        "rname",
        InsertFields(
          Ref("lname", leftElemType),
          FastSeq("intervals" -> Ref("rname", TArray(rightElemType))),
        ),
      )

    val prunedLElemType = leftElemType.deleteKey("b")
    val prunedRElemType = rightElemType.deleteKey("ignored")
    val requestedElemType = prunedLElemType.insertFields(
      FastSeq("intervals" -> TArray(prunedRElemType))
    )

    checkMemo(
      join,
      TStream(requestedElemType),
      Array(
        TStream(prunedLElemType),
        TStream(prunedRElemType),
        requestedElemType,
      ),
    )

    checkRebuild[StreamLeftIntervalJoin](
      join,
      TStream(requestedElemType),
      (_, pruned) =>
        pruned.left.typ == TStream(prunedLElemType) &&
          pruned.right.typ == TStream(prunedRElemType) &&
          pruned.body.typ == requestedElemType,
    )
  }

  @Test def testStreamForMemo(): Unit =
    checkMemo(
      StreamFor(st, "foo", Begin(FastSeq(GetField(Ref("foo", ref.typ), "a")))),
      TVoid,
      Array(TStream(justA), null),
    )

  @Test def testMakeNDArrayMemo(): Unit = {
    checkMemo(
      MakeNDArray(
        Ref("x", TArray(TStruct("a" -> TInt32, "b" -> TInt64))),
        Ref("y", TTuple(TInt32, TInt32)),
        True(),
        ErrorIDs.NO_ERROR,
      ),
      TNDArray(TStruct("a" -> TInt32), Nat(2)),
      Array(
        TArray(TStruct("a" -> TInt32)),
        TTuple(TInt32, TInt32),
        TBoolean,
      ),
    )
  }

  @Test def testNDArrayMapMemo(): Unit =
    checkMemo(
      NDArrayMap(ndArr, "foo", Ref("foo", ref.typ)),
      TNDArray(justBRequired, Nat(1)),
      Array(TNDArray(justBRequired, Nat(1)), null),
    )

  @Test def testNDArrayMap2Memo(): Unit = {
    checkMemo(
      NDArrayMap2(ndArr, ndArr, "left", "right", Ref("left", ref.typ), ErrorIDs.NO_ERROR),
      TNDArray(justBRequired, Nat(1)),
      Array(TNDArray(justBRequired, Nat(1)), TNDArray(TStruct.empty, Nat(1)), null),
    )
    checkMemo(
      NDArrayMap2(ndArr, ndArr, "left", "right", Ref("right", ref.typ), ErrorIDs.NO_ERROR),
      TNDArray(justBRequired, Nat(1)),
      Array(TNDArray(TStruct.empty, Nat(1)), TNDArray(justBRequired, Nat(1)), null),
    )
    val addFieldsIR = ApplyBinaryPrimOp(
      Add(),
      GetField(Ref("left", ref.typ), "a"),
      GetField(Ref("right", ref.typ), "b"),
    )
    checkMemo(
      NDArrayMap2(ndArr, ndArr, "left", "right", addFieldsIR, ErrorIDs.NO_ERROR),
      TNDArray(TInt32, Nat(1)),
      Array(TNDArray(justARequired, Nat(1)), TNDArray(justBRequired, Nat(1)), null),
    )
  }

  @Test def testMakeStructMemo(): Unit = {
    checkMemo(
      MakeStruct(IndexedSeq("a" -> ref, "b" -> I32(10))),
      TStruct("a" -> justA),
      Array(justA, null),
    )
    checkMemo(MakeStruct(IndexedSeq("a" -> ref, "b" -> I32(10))), TStruct.empty, Array(null, null))
  }

  @Test def testInsertFieldsMemo(): Unit =
    checkMemo(
      InsertFields(ref, IndexedSeq("d" -> ref)),
      justA ++ TStruct("d" -> justB),
      Array(justA, justB),
    )

  @Test def testSelectFieldsMemo(): Unit = {
    checkMemo(SelectFields(ref, IndexedSeq("a", "b")), justA, Array(justA))
    checkMemo(SelectFields(ref, IndexedSeq("b", "a")), bAndA, Array(aAndB))
  }

  @Test def testGetFieldMemo(): Unit =
    checkMemo(GetField(ref, "a"), TInt32, Array(justA))

  @Test def testMakeTupleMemo(): Unit =
    checkMemo(MakeTuple(IndexedSeq(0 -> ref)), TTuple(justA), Array(justA))

  @Test def testGetTupleElementMemo(): Unit =
    checkMemo(
      GetTupleElement(MakeTuple.ordered(IndexedSeq(ref, ref)), 1),
      justB,
      Array(TTuple(FastSeq(TupleField(1, justB)))),
    )

  @Test def testCastRenameMemo(): Unit = {
    checkMemo(
      CastRename(
        Ref("x", TArray(TStruct("x" -> TInt32, "y" -> TString))),
        TArray(TStruct("y" -> TInt32, "z" -> TString)),
      ),
      TArray(TStruct("z" -> TString)),
      Array(TArray(TStruct("y" -> TString))),
    )
  }

  @Test def testAggFilterMemo(): Unit = {
    val t = TStruct("a" -> TInt32, "b" -> TInt64, "c" -> TString)
    val select = SelectFields(Ref("x", t), IndexedSeq("c"))
    checkMemo(
      AggFilter(
        ApplyComparisonOp(LT(TInt32, TInt32), GetField(Ref("x", t), "a"), I32(0)),
        ApplyAggOp(
          FastSeq(),
          FastSeq(select),
          AggSignature(Collect(), FastSeq(), FastSeq(select.typ)),
        ),
        false,
      ),
      TArray(TStruct("c" -> TString)),
      Array(null, TArray(TStruct("c" -> TString))),
    )
  }

  @Test def testAggExplodeMemo(): Unit = {
    val t = TStream(TStruct("a" -> TInt32, "b" -> TInt64))
    val select = SelectFields(Ref("foo", t.elementType), IndexedSeq("a"))
    checkMemo(
      AggExplode(
        Ref("x", t),
        "foo",
        ApplyAggOp(
          FastSeq(),
          FastSeq(select),
          AggSignature(Collect(), FastSeq(), FastSeq(select.typ)),
        ),
        false,
      ),
      TArray(TStruct("a" -> TInt32)),
      Array(TStream(TStruct("a" -> TInt32)), TArray(TStruct("a" -> TInt32))),
    )
  }

  @Test def testAggArrayPerElementMemo(): Unit = {
    val t = TArray(TStruct("a" -> TInt32, "b" -> TInt64))
    val select = SelectFields(Ref("foo", t.elementType), IndexedSeq("a"))
    checkMemo(
      AggArrayPerElement(
        Ref("x", t),
        "foo",
        "bar",
        ApplyAggOp(
          FastSeq(),
          FastSeq(select),
          AggSignature(Collect(), FastSeq(), FastSeq(select.typ)),
        ),
        None,
        false,
      ),
      TArray(TArray(TStruct("a" -> TInt32))),
      Array(TArray(TStruct("a" -> TInt32)), TArray(TStruct("a" -> TInt32))),
    )
  }

  @Test def testCDAMemo(): Unit = {
    val ctxT = TStruct("a" -> TInt32, "b" -> TString)
    val globT = TStruct("c" -> TInt64, "d" -> TFloat64)
    val x = CollectDistributedArray(
      NA(TStream(ctxT)),
      NA(globT),
      "ctx",
      "glob",
      MakeTuple.ordered(FastSeq(Ref("ctx", ctxT), Ref("glob", globT))),
      NA(TString),
      "test",
    )

    checkMemo(
      x,
      TArray(TTuple(ctxT.typeAfterSelectNames(Array("a")), globT.typeAfterSelectNames(Array("c")))),
      Array(
        TStream(ctxT.typeAfterSelectNames(Array("a"))),
        globT.typeAfterSelectNames(Array("c")),
        null,
        TString,
      ),
    )
  }

  @Test def testTableCountMemo(): Unit =
    checkMemo(TableCount(tab), TInt64, Array(subsetTable(tab.typ, "NO_KEY")))

  @Test def testTableGetGlobalsMemo(): Unit =
    checkMemo(
      TableGetGlobals(tab),
      TStruct("g1" -> TInt32),
      Array(subsetTable(tab.typ, "global.g1", "NO_KEY")),
    )

  @Test def testTableCollectMemo(): Unit =
    checkMemo(
      TableCollect(tab),
      TStruct("rows" -> TArray(TStruct("3" -> TString)), "global" -> TStruct("g2" -> TInt32)),
      Array(subsetTable(tab.typ, "row.3", "global.g2")),
    )

  @Test def testTableHeadMemo(): Unit =
    checkMemo(
      TableHead(tab, 10L),
      subsetTable(tab.typ.copy(key = FastSeq()), "global.g1"),
      Array(subsetTable(tab.typ, "row.3", "global.g1")),
    )

  @Test def testTableTailMemo(): Unit =
    checkMemo(
      TableTail(tab, 10L),
      subsetTable(tab.typ.copy(key = FastSeq()), "global.g1"),
      Array(subsetTable(tab.typ, "row.3", "global.g1")),
    )

  @Test def testTableToValueApplyMemo(): Unit =
    checkMemo(
      TableToValueApply(tab, ForceCountTable()),
      TInt64,
      Array(tab.typ),
    )

  @Test def testMatrixToValueApplyMemo(): Unit =
    checkMemo(
      MatrixToValueApply(mat, ForceCountMatrixTable()),
      TInt64,
      Array(mat.typ),
    )

  @Test def testTableAggregateMemo(): Unit =
    checkMemo(
      TableAggregate(tab, tableRefBoolean(tab.typ, "global.g1")),
      TBoolean,
      Array(subsetTable(tab.typ, "global.g1"), null),
    )

  @Test def testMatrixAggregateMemo(): Unit =
    checkMemo(
      MatrixAggregate(mat, matrixRefBoolean(mat.typ, "global.g1")),
      TBoolean,
      Array(subsetMatrixTable(mat.typ, "global.g1", "NO_COL_KEY"), null),
    )

  @Test def testPipelineLetMemo(): Unit = {
    val t = TStruct("a" -> TInt32)
    checkMemo(
      RelationalLet("foo", NA(t), RelationalRef("foo", t)),
      TStruct.empty,
      Array(TStruct.empty, TStruct.empty),
    )
  }

  @Test def testTableFilterRebuild(): Unit = {
    checkRebuild(
      TableFilter(tr, tableRefBoolean(tr.typ, "row.2")),
      subsetTable(tr.typ, "row.3"),
      (_: BaseIR, r: BaseIR) => {
        val tf = r.asInstanceOf[TableFilter]
        TypeCheck(ctx, tf.pred, PruneDeadFields.relationalTypeToEnv(tf.typ))
        tf.child.typ == subsetTable(tr.typ, "row.3", "row.2")
      },
    )
  }

  @Test def testTableMapRowsRebuild(): Unit = {
    val tmr = TableMapRows(tr, tableRefStruct(tr.typ, "row.2", "global.g1"))
    checkRebuild(
      tmr,
      subsetTable(tmr.typ, "row.foo"),
      (_: BaseIR, r: BaseIR) => {
        val tmr = r.asInstanceOf[TableMapRows]
        TypeCheck(ctx, tmr.newRow, PruneDeadFields.relationalTypeToEnv(tmr.child.typ))
        tmr.child.typ == subsetTable(tr.typ, "row.2", "global.g1", "row.3")
      },
    )

    val tmr2 = TableMapRows(tr, tableRefStruct(tr.typ, "row.2", "global.g1"))
    checkRebuild(
      tmr2,
      subsetTable(tmr2.typ, "row.foo", "NO_KEY"),
      (_: BaseIR, r: BaseIR) => {
        val tmr = r.asInstanceOf[TableMapRows]
        TypeCheck(ctx, tmr.newRow, PruneDeadFields.relationalTypeToEnv(tmr.child.typ))
        tmr.child.typ == subsetTable(
          tr.typ,
          "row.2",
          "global.g1",
          "row.3",
          "NO_KEY",
        ) // FIXME: remove row.3 when TableRead is fixed
      },
    )

  }

  @Test def testTableMapGlobalsRebuild(): Unit = {
    val tmg = TableMapGlobals(tr, tableRefStruct(tr.typ, "global.g1"))
    checkRebuild(
      tmg,
      subsetTable(tmg.typ, "global.foo"),
      (_: BaseIR, r: BaseIR) => {
        val tmg = r.asInstanceOf[TableMapGlobals]
        TypeCheck(ctx, tmg.newGlobals, PruneDeadFields.relationalTypeToEnv(tmg.child.typ))
        tmg.child.typ == subsetTable(tr.typ, "global.g1")
      },
    )
  }

  @Test def testTableLeftJoinRightDistinctRebuild(): Unit = {
    val tk1 = TableKeyBy(tab, Array("1"))
    val tk2 = TableKeyBy(tab, Array("3"))
    val tj = TableLeftJoinRightDistinct(tk1, tk2, "foo")

    checkRebuild(
      tj,
      subsetTable(tj.typ, "row.1", "row.4"),
      (_: BaseIR, r: BaseIR) =>
        r.isInstanceOf[TableKeyBy], // no dependence on row.foo elides the join
    )
  }

  @Test def testTableIntervalJoinRebuild(): Unit = {
    val tk1 = TableKeyBy(tab, Array("1"))
    val tk2 = TableKeyBy(tab, Array("3"))
    val tj = TableIntervalJoin(tk1, tk2, "foo", product = false)

    checkRebuild(
      tj,
      subsetTable(tj.typ, "row.1", "row.4"),
      (_: BaseIR, r: BaseIR) =>
        r.isInstanceOf[TableKeyBy], // no dependence on row.foo elides the join
    )
  }

  @Test def testTableUnionRebuildUnifiesRowTypes(): Unit = {
    val mapExpr = InsertFields(
      Ref("row", tr.typ.rowType),
      FastSeq("foo" -> tableRefBoolean(tr.typ, "row.3", "global.g1")),
    )
    val tfilter = TableFilter(
      TableMapRows(tr, mapExpr),
      tableRefBoolean(tr.typ, "row.2"),
    )
    val tmap = TableMapRows(tr, mapExpr)
    val tunion = TableUnion(FastSeq(tfilter, tmap))
    checkRebuild(
      tunion,
      subsetTable(tunion.typ, "row.foo"),
      (_: BaseIR, rebuilt: BaseIR) => {
        val tu = rebuilt.asInstanceOf[TableUnion]
        val tf = tu.childrenSeq(0)
        val tm = tu.childrenSeq(1)
        tf.typ.rowType == tm.typ.rowType &&
        tu.typ == subsetTable(tunion.typ, "row.foo", "global.g1")
      },
    )
  }

  @Test def testTableMultiWayZipJoinRebuildUnifiesRowTypes(): Unit = {
    val t1 = TableKeyBy(tab, Array("1"))
    val t2 = TableFilter(t1, tableRefBoolean(t1.typ, "row.2"))
    val t3 = TableFilter(t1, tableRefBoolean(t1.typ, "row.3"))
    val ts = Array(t1, t2, t3)
    val tmwzj = TableMultiWayZipJoin(ts, "data", "gbls")
    val childRType = subsetTable(t1.typ, "row.2", "global.g1")
    checkRebuild(
      tmwzj,
      subsetTable(tmwzj.typ, "row.data.2", "global.gbls.g1"),
      (_: BaseIR, rebuilt: BaseIR) => {
        val t = rebuilt.asInstanceOf[TableMultiWayZipJoin]
        t.childrenSeq.forall(c => c.typ == childRType)
      },
    )
  }

  @Test def testMatrixFilterColsRebuild(): Unit = {
    val mfc = MatrixFilterCols(mr, matrixRefBoolean(mr.typ, "sa.c2"))
    checkRebuild(
      mfc,
      subsetMatrixTable(mfc.typ, "global.g1"),
      (_: BaseIR, r: BaseIR) => {
        val mfc = r.asInstanceOf[MatrixFilterCols]
        TypeCheck(ctx, mfc.pred, PruneDeadFields.relationalTypeToEnv(mfc.child.typ))
        mfc.child.asInstanceOf[MatrixRead].typ == subsetMatrixTable(mr.typ, "global.g1", "sa.c2")
      },
    )
  }

  @Test def testMatrixFilterEntriesRebuild(): Unit = {
    val mfe = MatrixFilterEntries(mr, matrixRefBoolean(mr.typ, "sa.c2", "va.r2", "g.e1"))
    checkRebuild(
      mfe,
      subsetMatrixTable(mfe.typ, "global.g1"),
      (_: BaseIR, r: BaseIR) => {
        val mfe = r.asInstanceOf[MatrixFilterEntries]
        TypeCheck(ctx, mfe.pred, PruneDeadFields.relationalTypeToEnv(mfe.child.typ))
        mfe.child.asInstanceOf[MatrixRead].typ == subsetMatrixTable(
          mr.typ,
          "global.g1",
          "sa.c2",
          "va.r2",
          "g.e1",
        )
      },
    )
  }

  @Test def testMatrixMapRowsRebuild(): Unit = {
    val mmr = MatrixMapRows(
      MatrixKeyRowsBy(mr, IndexedSeq.empty),
      matrixRefStruct(mr.typ, "va.r2"),
    )
    checkRebuild(
      mmr,
      subsetMatrixTable(mmr.typ, "global.g1", "g.e1", "va.foo"),
      (_: BaseIR, r: BaseIR) => {
        val mmr = r.asInstanceOf[MatrixMapRows]
        TypeCheck(ctx, mmr.newRow, PruneDeadFields.relationalTypeToEnv(mmr.child.typ))
        mmr.child.asInstanceOf[MatrixKeyRowsBy].child.asInstanceOf[
          MatrixRead
        ].typ == subsetMatrixTable(mr.typ, "global.g1", "va.r2", "g.e1")
      },
    )
  }

  @Test def testMatrixMapColsRebuild(): Unit = {
    val mmc = MatrixMapCols(mr, matrixRefStruct(mr.typ, "sa.c2"), Some(FastSeq("foo")))
    checkRebuild(
      mmc,
      subsetMatrixTable(mmc.typ, "global.g1", "g.e1", "sa.foo"),
      (_: BaseIR, r: BaseIR) => {
        val mmc = r.asInstanceOf[MatrixMapCols]
        TypeCheck(ctx, mmc.newCol, PruneDeadFields.relationalTypeToEnv(mmc.child.typ))
        mmc.child.asInstanceOf[MatrixRead].typ == subsetMatrixTable(
          mr.typ,
          "global.g1",
          "sa.c2",
          "g.e1",
        )
      },
    )
  }

  @Test def testMatrixMapEntriesRebuild(): Unit = {
    val mme = MatrixMapEntries(mr, matrixRefStruct(mr.typ, "sa.c2", "va.r2"))
    checkRebuild(
      mme,
      subsetMatrixTable(mme.typ, "global.g1", "g.foo"),
      (_: BaseIR, r: BaseIR) => {
        val mme = r.asInstanceOf[MatrixMapEntries]
        TypeCheck(ctx, mme.newEntries, PruneDeadFields.relationalTypeToEnv(mme.child.typ))
        mme.child.asInstanceOf[MatrixRead].typ == subsetMatrixTable(
          mr.typ,
          "global.g1",
          "sa.c2",
          "va.r2",
        )
      },
    )
  }

  @Test def testMatrixMapGlobalsRebuild(): Unit = {
    val mmg = MatrixMapGlobals(mr, matrixRefStruct(mr.typ, "global.g1"))
    checkRebuild(
      mmg,
      subsetMatrixTable(mmg.typ, "global.foo", "g.e1", "va.r2"),
      (_: BaseIR, r: BaseIR) => {
        val mmg = r.asInstanceOf[MatrixMapGlobals]
        TypeCheck(ctx, mmg.newGlobals, PruneDeadFields.relationalTypeToEnv(mmg.child.typ))
        mmg.child.asInstanceOf[MatrixRead].typ == subsetMatrixTable(
          mr.typ,
          "global.g1",
          "va.r2",
          "g.e1",
        )
      },
    )
  }

  @Test def testMatrixAggregateRowsByKeyRebuild(): Unit = {
    val ma = MatrixAggregateRowsByKey(
      mr,
      matrixRefStruct(mr.typ, "sa.c2"),
      matrixRefStruct(mr.typ, "global.g1"),
    )
    checkRebuild(
      ma,
      subsetMatrixTable(ma.typ, "va.foo", "g.foo"),
      (_: BaseIR, r: BaseIR) => {
        val ma = r.asInstanceOf[MatrixAggregateRowsByKey]
        TypeCheck(ctx, ma.entryExpr, PruneDeadFields.relationalTypeToEnv(ma.child.typ))
        ma.child.asInstanceOf[MatrixRead].typ == subsetMatrixTable(mr.typ, "global.g1", "sa.c2")
      },
    )
  }

  @Test def testMatrixAggregateColsByKeyRebuild(): Unit = {
    val ma = MatrixAggregateColsByKey(
      mr,
      matrixRefStruct(mr.typ, "va.r2"),
      matrixRefStruct(mr.typ, "global.g1"),
    )
    checkRebuild(
      ma,
      subsetMatrixTable(ma.typ, "g.foo", "sa.foo"),
      (_: BaseIR, r: BaseIR) => {
        val ma = r.asInstanceOf[MatrixAggregateColsByKey]
        TypeCheck(ctx, ma.entryExpr, PruneDeadFields.relationalTypeToEnv(ma.child.typ))
        ma.child.asInstanceOf[MatrixRead].typ == subsetMatrixTable(mr.typ, "global.g1", "va.r2")
      },
    )
  }

  @Test def testMatrixUnionRowsRebuild(): Unit = {
    val mat2 = MatrixLiteral(mType.copy(colKey = FastSeq()), mat.tl)
    checkRebuild(
      MatrixUnionRows(FastSeq(
        mat,
        MatrixMapCols(mat2, Ref("sa", mat2.typ.colType), Some(FastSeq("ck"))),
      )),
      mat.typ.copy(colKey = FastSeq()),
      (_: BaseIR, r: BaseIR) =>
        r.asInstanceOf[MatrixUnionRows].childrenSeq.forall {
          _.typ.colKey.isEmpty
        },
    )
  }

  @Test def testMatrixUnionColsRebuild(): Unit = {
    def getColField(name: String) =
      GetField(Ref("sa", mat.typ.colType), name)
    def childrenMatch(matrixUnionCols: MatrixUnionCols): Boolean =
      matrixUnionCols.left.typ.colType == matrixUnionCols.right.typ.colType &&
        matrixUnionCols.left.typ.entryType == matrixUnionCols.right.typ.entryType

    val wrappedMat = MatrixMapCols(
      mat,
      MakeStruct(IndexedSeq(
        ("ck", getColField("ck")),
        ("c2", getColField("c2")),
        ("c3", getColField("c3")),
      )),
      Some(FastSeq("ck")),
    )

    val wrappedMat2 = MatrixRename(
      wrappedMat,
      Map.empty,
      Map.empty,
      wrappedMat.typ.rowType.fieldNames.map(x => x -> (x + "_")).toMap,
      Map.empty,
    )

    val mucBothSame = MatrixUnionCols(wrappedMat, wrappedMat2, "inner")
    checkRebuild(mucBothSame, mucBothSame.typ)
    checkRebuild[MatrixUnionCols](
      mucBothSame,
      mucBothSame.typ.copy(colType = TStruct(("ck", TString), ("c2", TInt32))),
      (old, rebuilt) =>
        (old.typ.rowType == rebuilt.typ.rowType) &&
          (old.typ.globalType == rebuilt.typ.globalType) &&
          (rebuilt.typ.colType.fieldNames.toIndexedSeq == IndexedSeq("ck", "c2")) &&
          childrenMatch(rebuilt),
    )

    /* Since `mat` is a MatrixLiteral, it won't be rebuilt, will keep all fields. But wrappedMat is
     * a MatrixMapCols, so it will drop */
    /* unrequested fields. This test would fail without upcasting in the MatrixUnionCols rebuild
     * rule. */
    val muc2 = MatrixUnionCols(mat, wrappedMat2, "inner")
    checkRebuild[MatrixUnionCols](
      muc2,
      muc2.typ.copy(colType = TStruct(("ck", TString))),
      (old, rebuilt) =>
        childrenMatch(rebuilt),
    )

  }

  @Test def testMatrixAnnotateRowsTableRebuild(): Unit = {
    val tl = TableLiteral(Interpret(MatrixRowsTable(mat), ctx), theHailClassLoader)
    val mart = MatrixAnnotateRowsTable(mat, tl, "foo", product = false)
    checkRebuild(
      mart,
      subsetMatrixTable(mart.typ),
      (_: BaseIR, r: BaseIR) =>
        r.isInstanceOf[MatrixLiteral],
    )
  }

  val ts = TStruct(
    "a" -> TInt32,
    "b" -> TInt64,
    "c" -> TString,
  )

  def subsetTS(fields: String*): TStruct = ts.filterSet(fields.toSet)._1

  @Test def testNARebuild(): Unit = {
    checkRebuild(
      NA(ts),
      subsetTS("b"),
      (_: BaseIR, r: BaseIR) => {
        val na = r.asInstanceOf[NA]
        na.typ == subsetTS("b")
      },
    )
  }

  @Test def testIfRebuild(): Unit = {
    checkRebuild(
      If(True(), NA(ts), NA(ts)),
      subsetTS("b"),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[If]
        ir.cnsq.typ == subsetTS("b") && ir.altr.typ == subsetTS("b")
      },
    )
  }

  @Test def testSwitchRebuild(): Unit =
    checkRebuild[IR](
      Switch(I32(0), NA(ts), FastSeq(NA(ts))),
      subsetTS("b"),
      {
        case (_, Switch(_, default, cases)) =>
          default.typ == subsetTS("b") &&
          cases(0).typ == subsetTS("b")
      },
    )

  @Test def testCoalesceRebuild(): Unit = {
    checkRebuild(
      Coalesce(FastSeq(NA(ts), NA(ts))),
      subsetTS("b"),
      (_: BaseIR, r: BaseIR) =>
        r.children.forall(_.typ == subsetTS("b")),
    )
  }

  @Test def testLetRebuild(): Unit = {
    checkRebuild(
      Let(FastSeq("x" -> NA(ts)), Ref("x", ts)),
      subsetTS("b"),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[Block]
        ir.bindings.head.value.typ == subsetTS("b")
      },
    )
  }

  @Test def testAggLetRebuild(): Unit = {
    checkRebuild(
      AggLet(
        "foo",
        NA(ref.typ),
        ApplyAggOp(
          FastSeq(),
          FastSeq(
            SelectFields(Ref("foo", ref.typ), IndexedSeq("a"))
          ),
          AggSignature(Collect(), FastSeq(), FastSeq(ref.typ)),
        ),
        false,
      ),
      TArray(subsetTS("a")),
      (_: BaseIR, r: BaseIR) =>
        r match {
          case Block(Seq(Binding(_, value, Scope.AGG)), _) =>
            value.typ == subsetTS("a")
        },
    )
  }

  @Test def testMakeArrayRebuild(): Unit = {
    checkRebuild(
      MakeArray(IndexedSeq(NA(ts)), TArray(ts)),
      TArray(subsetTS("b")),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[MakeArray]
        ir.args.head.typ == subsetTS("b")
      },
    )
  }

  @Test def testStreamTakeRebuild(): Unit = {
    checkRebuild(
      StreamTake(MakeStream(IndexedSeq(NA(ts)), TStream(ts)), I32(2)),
      TStream(subsetTS("b")),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[StreamTake]
        ir.a.typ == TStream(subsetTS("b"))
      },
    )
  }

  @Test def testStreamDropRebuild(): Unit = {
    checkRebuild(
      StreamDrop(MakeStream(IndexedSeq(NA(ts)), TStream(ts)), I32(2)),
      TStream(subsetTS("b")),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[StreamDrop]
        ir.a.typ == TStream(subsetTS("b"))
      },
    )
  }

  @Test def testStreamMapRebuild(): Unit = {
    checkRebuild(
      StreamMap(MakeStream(IndexedSeq(NA(ts)), TStream(ts)), "x", Ref("x", ts)),
      TStream(subsetTS("b")),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[StreamMap]
        ir.a.typ == TStream(subsetTS("b"))
      },
    )
  }

  @Test def testStreamGroupedRebuild(): Unit = {
    checkRebuild(
      StreamGrouped(MakeStream(IndexedSeq(NA(ts)), TStream(ts)), I32(2)),
      TStream(TStream(subsetTS("b"))),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[StreamGrouped]
        ir.a.typ == TStream(subsetTS("b"))
      },
    )
  }

  @Test def testStreamGroupByKeyRebuild(): Unit = {
    checkRebuild(
      StreamGroupByKey(MakeStream(IndexedSeq(NA(ts)), TStream(ts)), FastSeq("a"), false),
      TStream(TStream(subsetTS("b"))),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[StreamGroupByKey]
        ir.a.typ == TStream(subsetTS("a", "b"))
      },
    )
  }

  @Test def testStreamMergeRebuild(): Unit = {
    checkRebuild(
      StreamMultiMerge(
        IndexedSeq(
          MakeStream(IndexedSeq(NA(ts)), TStream(ts)),
          MakeStream(IndexedSeq(NA(ts)), TStream(ts)),
        ),
        FastSeq("a"),
      ),
      TStream(subsetTS("b")),
      (_: BaseIR, r: BaseIR) => r.typ == TStream(subsetTS("a", "b")),
    )
  }

  @Test def testStreamZipRebuild(): Unit = {
    val a2 = st.deepCopy()
    val a3 = st.deepCopy()
    for (
      b <- Array(
        ArrayZipBehavior.ExtendNA,
        ArrayZipBehavior.TakeMinLength,
        ArrayZipBehavior.AssertSameLength,
      )
    ) {

      checkRebuild(
        StreamZip(
          FastSeq(st, a2, a3),
          FastSeq("foo", "bar", "baz"),
          Let(
            FastSeq(
              "foo1" -> GetField(Ref("foo", ref.typ), "b"),
              "bar2" -> GetField(Ref("bar", ref.typ), "a"),
            ),
            False(),
          ),
          b,
        ),
        TStream(TBoolean),
        (_: BaseIR, r: BaseIR) => r.asInstanceOf[StreamZip].as.length == 3,
      )
    }
    checkRebuild(
      StreamZip(
        FastSeq(st, a2, a3),
        FastSeq("foo", "bar", "baz"),
        Let(
          FastSeq(
            "foo1" -> GetField(Ref("foo", ref.typ), "b"),
            "bar2" -> GetField(Ref("bar", ref.typ), "a"),
          ),
          False(),
        ),
        ArrayZipBehavior.AssumeSameLength,
      ),
      TStream(TBoolean),
      (_: BaseIR, r: BaseIR) => r.asInstanceOf[StreamZip].as.length == 2,
    )
  }

  @Test def testStreamFlatmapRebuild(): Unit = {
    checkRebuild(
      StreamFlatMap(
        MakeStream(IndexedSeq(NA(ts)), TStream(ts)),
        "x",
        MakeStream(IndexedSeq(Ref("x", ts)), TStream(ts)),
      ),
      TStream(subsetTS("b")),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[StreamFlatMap]
        ir.a.typ == TStream(subsetTS("b"))
      },
    )
  }

  @Test def testMakeStructRebuild(): Unit = {
    checkRebuild(
      MakeStruct(IndexedSeq("a" -> NA(TInt32), "b" -> NA(TInt64), "c" -> NA(TString))),
      subsetTS("b"),
      (_: BaseIR, r: BaseIR) =>
        r == MakeStruct(IndexedSeq("b" -> NA(TInt64))),
    )
  }

  @Test def testInsertFieldsRebuild(): Unit = {
    checkRebuild(
      InsertFields(NA(TStruct("a" -> TInt32)), IndexedSeq("b" -> NA(TInt64), "c" -> NA(TString))),
      subsetTS("b"),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[InsertFields]
        ir.fields == IndexedSeq(
          "b" -> NA(TInt64)
        )
      },
    )

    /* Example needs to have field insertion that overwrites an unrequested field with a different
     * type. */
    val insertF =
      InsertFields(Ref("foo", TStruct(("a", TInt32), ("b", TInt32))), IndexedSeq(("a", I64(8))))
    checkRebuild[InsertFields](
      insertF,
      TStruct(("b", TInt32)),
      (old, rebuilt) =>
        PruneDeadFields.isSupertype(rebuilt.typ, old.typ),
    )
  }

  @Test def testMakeTupleRebuild(): Unit = {
    checkRebuild(
      MakeTuple(IndexedSeq(0 -> I32(1), 1 -> F64(1.0), 2 -> NA(TString))),
      TTuple(FastSeq(TupleField(2, TString))),
      (_: BaseIR, r: BaseIR) =>
        r == MakeTuple(IndexedSeq(2 -> NA(TString))),
    )
  }

  @Test def testSelectFieldsRebuild(): Unit = {
    checkRebuild(
      SelectFields(NA(ts), IndexedSeq("a", "b")),
      subsetTS("b"),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[SelectFields]
        ir.fields == IndexedSeq("b")
      },
    )
  }

  @Test def testCastRenameRebuild(): Unit = {
    checkRebuild(
      CastRename(
        NA(TArray(TStruct("x" -> TInt32, "y" -> TString))),
        TArray(TStruct("y" -> TInt32, "z" -> TString)),
      ),
      TArray(TStruct("z" -> TString)),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[CastRename]
        ir._typ == TArray(TStruct("z" -> TString))
      },
    )
  }

  val ndArrayTS = MakeNDArray(
    MakeArray(ArrayBuffer(NA(ts)), TArray(ts)),
    MakeTuple(IndexedSeq((0, I64(1L)))),
    True(),
    ErrorIDs.NO_ERROR,
  )

  @Test def testNDArrayMapRebuild(): Unit = {
    checkRebuild(
      NDArrayMap(ndArrayTS, "x", Ref("x", ts)),
      TNDArray(subsetTS("b"), Nat(1)),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[NDArrayMap]
        /* Even though the type I requested wasn't required, NDArrays always have a required element
         * type. */
        ir.nd.typ == TNDArray(TStruct(("b", TInt64)), Nat(1))
      },
    )
  }

  @Test def testNDArrayMap2Rebuild(): Unit = {
    checkRebuild(
      NDArrayMap2(ndArrayTS, ndArrayTS, "left", "right", Ref("left", ts), ErrorIDs.NO_ERROR),
      TNDArray(subsetTS("b"), Nat(1)),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[NDArrayMap2]
        ir.l.typ == TNDArray(TStruct(("b", TInt64)), Nat(1))
        ir.r.typ == TNDArray(TStruct.empty, Nat(1))
      },
    )
    checkRebuild(
      NDArrayMap2(ndArrayTS, ndArrayTS, "left", "right", Ref("right", ts), ErrorIDs.NO_ERROR),
      TNDArray(subsetTS("b"), Nat(1)),
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[NDArrayMap2]
        ir.l.typ == TNDArray(TStruct.empty, Nat(1))
        ir.r.typ == TNDArray(TStruct(("b", TInt64)), Nat(1))
      },
    )
  }

  @Test def testCDARebuild(): Unit = {
    val ctxT = TStruct("a" -> TInt32, "b" -> TString)
    val globT = TStruct("c" -> TInt64, "d" -> TFloat64)
    val x = CollectDistributedArray(
      NA(TStream(ctxT)),
      NA(globT),
      "ctx",
      "glob",
      MakeTuple.ordered(FastSeq(Ref("ctx", ctxT), Ref("glob", globT))),
      NA(TString),
      "test",
    )

    val selectedCtxT = ctxT.typeAfterSelectNames(Array("a"))
    val selectedGlobT = globT.typeAfterSelectNames(Array("c"))
    checkRebuild(
      x,
      TArray(TTuple(selectedCtxT, selectedGlobT)),
      (_: BaseIR, r: BaseIR) => {
        r == CollectDistributedArray(
          NA(TStream(selectedCtxT)),
          NA(selectedGlobT),
          "ctx",
          "glob",
          MakeTuple.ordered(FastSeq(Ref("ctx", selectedCtxT), Ref("glob", selectedGlobT))),
          NA(TString),
          "test",
        )
      },
    )
  }

  @Test def testTableAggregateRebuild(): Unit = {
    val ta = TableAggregate(tr, tableRefBoolean(tr.typ, "row.2"))
    checkRebuild(
      ta,
      TBoolean,
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[TableAggregate]
        ir.child.typ == subsetTable(tr.typ, "row.2")
      },
    )
  }

  @Test def testTableCollectRebuild(): Unit = {
    val tc = TableCollect(TableKeyBy(tab, FastSeq()))
    checkRebuild(
      tc,
      TStruct("global" -> TStruct("g1" -> TInt32)),
      (_: BaseIR, r: BaseIR) =>
        r.asInstanceOf[MakeStruct].fields.head._2.isInstanceOf[TableGetGlobals],
    )

    checkRebuild(
      tc,
      TStruct.empty,
      (_: BaseIR, r: BaseIR) =>
        r == MakeStruct(IndexedSeq()),
    )
  }

  @Test def testMatrixAggregateRebuild(): Unit = {
    val ma = MatrixAggregate(mr, matrixRefBoolean(mr.typ, "va.r2"))
    checkRebuild(
      ma,
      TBoolean,
      (_: BaseIR, r: BaseIR) => {
        val ir = r.asInstanceOf[MatrixAggregate]
        ir.child.typ == subsetMatrixTable(mr.typ, "va.r2")
      },
    )
  }

  @Test def testPipelineLetRebuild(): Unit = {
    val t = TStruct("a" -> TInt32)
    checkRebuild(
      RelationalLet("foo", NA(t), RelationalRef("foo", t)),
      TStruct.empty,
      (_: BaseIR, r: BaseIR) =>
        r.asInstanceOf[RelationalLet].body == RelationalRef("foo", TStruct.empty),
    )
  }

  @Test def testPipelineLetTableRebuild(): Unit = {
    val t = TStruct("a" -> TInt32)
    checkRebuild(
      RelationalLetTable("foo", NA(t), TableMapGlobals(tab, RelationalRef("foo", t))),
      tab.typ.copy(globalType = TStruct.empty),
      (_: BaseIR, r: BaseIR) =>
        r.asInstanceOf[RelationalLetTable].body.asInstanceOf[
          TableMapGlobals
        ].newGlobals == RelationalRef("foo", TStruct.empty),
    )
  }

  @Test def testPipelineLetMatrixTableRebuild(): Unit = {
    val t = TStruct("a" -> TInt32)
    checkRebuild(
      RelationalLetMatrixTable("foo", NA(t), MatrixMapGlobals(mat, RelationalRef("foo", t))),
      mat.typ.copy(globalType = TStruct.empty),
      (_: BaseIR, r: BaseIR) =>
        r.asInstanceOf[RelationalLetMatrixTable].body.asInstanceOf[
          MatrixMapGlobals
        ].newGlobals == RelationalRef("foo", TStruct.empty),
    )
  }

  @Test def testIfUnification(): Unit = {
    val pred = False()
    val t = TStruct("a" -> TInt32, "b" -> TInt32)
    val pruneT = TStruct("a" -> TInt32)
    val cnsq = Ref("x", t)
    val altr = NA(t)
    val ifIR = If(pred, cnsq, altr)
    val memo = Memo.empty[BaseType]
      .bind(pred, TBoolean)
      .bind(cnsq, pruneT)
      .bind(altr, pruneT)
      .bind(ifIR, pruneT)

    // should run without error!
    PruneDeadFields.rebuildIR(
      ctx,
      ifIR,
      BindingEnv.empty[Type].bindEval("a", t),
      PruneDeadFields.RebuildMutableState(memo, mutable.HashMap.empty),
    )
  }

  @DataProvider(name = "supertypePairs")
  def supertypePairs: Array[Array[Type]] = Array(
    Array(TInt32, TInt32),
    Array(
      TStruct(
        "a" -> TInt32,
        "b" -> TArray(TInt64),
      ),
      TStruct(
        "a" -> TInt32,
        "b" -> TArray(TInt64),
      ),
    ),
    Array(TSet(TString), TSet(TString)),
  )

  @Test(dataProvider = "supertypePairs")
  def testIsSupertypeRequiredness(t1: Type, t2: Type) =
    assert(
      PruneDeadFields.isSupertype(t1, t2),
      s"""Failure, supertype relationship not met
         | supertype: ${t1.toPrettyString(true)}
         | subtype:   ${t2.toPrettyString(true)}""".stripMargin,
    )

  @Test def testApplyScanOp(): Unit = {
    val x = Ref("x", TInt32)
    val y = Ref("y", TInt32)
    val collectScan = ApplyScanOp(
      FastSeq(),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(Collect(), FastSeq(), FastSeq(TStruct("x" -> TInt32, "y" -> TInt32))),
    )
    checkRebuild(
      collectScan,
      TArray(TStruct("y" -> TInt32)),
      (_: BaseIR, reb: BaseIR) => reb.typ == TArray(TStruct("y" -> TInt32)),
    )

    val takeScan = ApplyScanOp(
      FastSeq(I32(1)),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(Take(), FastSeq(TInt32), FastSeq(TStruct("x" -> TInt32, "y" -> TInt32))),
    )
    checkRebuild(
      takeScan,
      TArray(TStruct("y" -> TInt32)),
      (_: BaseIR, reb: BaseIR) => reb.typ == TArray(TStruct("y" -> TInt32)),
    )

    val prevnn = ApplyScanOp(
      FastSeq(),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(PrevNonnull(), FastSeq(), FastSeq(TStruct("x" -> TInt32, "y" -> TInt32))),
    )
    checkRebuild(
      prevnn,
      TStruct("y" -> TInt32),
      (_: BaseIR, reb: BaseIR) => reb.typ == TStruct("y" -> TInt32),
    )

    val takeByScan = ApplyScanOp(
      FastSeq(I32(1)),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y))), MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(
        TakeBy(),
        FastSeq(TInt32),
        FastSeq(TStruct("x" -> TInt32, "y" -> TInt32), TStruct("x" -> TInt32, "y" -> TInt32)),
      ),
    )
    checkRebuild(
      takeByScan,
      TArray(TStruct("y" -> TInt32)),
      { (_: BaseIR, reb: BaseIR) =>
        val s = reb.asInstanceOf[ApplyScanOp]
        s.seqOpArgs == FastSeq(
          MakeStruct(FastSeq(("y", y))),
          MakeStruct(FastSeq(("x", x), ("y", y))),
        )
      },
    )
  }

  @Test def testApplyAggOp(): Unit = {
    val x = Ref("x", TInt32)
    val y = Ref("y", TInt32)
    val collectAgg = ApplyAggOp(
      FastSeq(),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(Collect(), FastSeq(), FastSeq(TStruct("x" -> TInt32, "y" -> TInt32))),
    )
    checkRebuild(
      collectAgg,
      TArray(TStruct("y" -> TInt32)),
      (_: BaseIR, reb: BaseIR) => reb.typ == TArray(TStruct("y" -> TInt32)),
    )

    val takeAgg = ApplyAggOp(
      FastSeq(I32(1)),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(Take(), FastSeq(TInt32), FastSeq(TStruct("x" -> TInt32, "y" -> TInt32))),
    )
    checkRebuild(
      takeAgg,
      TArray(TStruct("y" -> TInt32)),
      (_: BaseIR, reb: BaseIR) => reb.typ == TArray(TStruct("y" -> TInt32)),
    )

    val prevnn = ApplyAggOp(
      FastSeq(),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(PrevNonnull(), FastSeq(), FastSeq(TStruct("x" -> TInt32, "y" -> TInt32))),
    )
    checkRebuild(
      prevnn,
      TStruct("y" -> TInt32),
      (_: BaseIR, reb: BaseIR) => reb.typ == TStruct("y" -> TInt32),
    )

    val takeByAgg = ApplyAggOp(
      FastSeq(I32(1)),
      FastSeq(MakeStruct(FastSeq(("x", x), ("y", y))), MakeStruct(FastSeq(("x", x), ("y", y)))),
      AggSignature(
        TakeBy(),
        FastSeq(TInt32),
        FastSeq(TStruct("x" -> TInt32, "y" -> TInt32), TStruct("x" -> TInt32, "y" -> TInt32)),
      ),
    )
    checkRebuild(
      takeByAgg,
      TArray(TStruct("y" -> TInt32)),
      { (_: BaseIR, reb: BaseIR) =>
        val a = reb.asInstanceOf[ApplyAggOp]
        a.seqOpArgs == FastSeq(
          MakeStruct(FastSeq(("y", y))),
          MakeStruct(FastSeq(("x", x), ("y", y))),
        )
      },
    )
  }

  @Test def testStreamFold2(): Unit = {
    val eltType = TStruct("a" -> TInt32, "b" -> TInt32)
    val accum1Type = TStruct("c" -> TInt32, "d" -> TInt32)

    val ir0 = StreamFold2(
      NA(TStream(eltType)),
      FastSeq("1" -> NA(accum1Type)),
      "elt",
      FastSeq(
        MakeStruct(FastSeq(
          "c" -> GetField(Ref("elt", eltType), "a"),
          "d" -> GetField(Ref("1", accum1Type), "c"),
        ))
      ),
      Ref("1", TStruct("c" -> TInt32, "d" -> TInt32)),
    )

    def checker(original: IR, rebuilt: IR): Boolean = {
      val r = rebuilt.asInstanceOf[StreamFold2]
      r.typ == TStruct("c" -> TInt32)
      r.a.typ == TStream(TStruct("a" -> TInt32))
      r.accum(0)._2.typ == r.typ
    }

    checkRebuild(ir0, TStruct("c" -> TInt32), checker)
  }
}
