package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PruneSuite extends SparkSuite {
  @Test def testUnionType() {
    val base = TStruct(
      "a" -> TStruct(
        "aa" -> TInt32(),
        "ab" -> TStruct(
          "aaa" -> TString())),
      "b" -> TInt32(),
      "c" -> TArray(TStruct(
        "ca" -> TInt32())))

    assert(PruneDeadFields.unify(base, TStruct()) == TStruct())
    assert(PruneDeadFields.unify(base, TStruct("b" -> TInt32())) == TStruct("b" -> TInt32()))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct())) == TStruct("a" -> TStruct()))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct()),
      TStruct("b" -> TInt32())) == TStruct("a" -> TStruct(), "b" -> TInt32()))
    assert(PruneDeadFields.unify(base, TStruct("c" -> TArray(TStruct()))) == TStruct("c" -> TArray(TStruct())))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct("ab" -> TStruct())),
      TStruct("c" -> TArray(TStruct()))) == TStruct("a" -> TStruct("ab" -> TStruct()), "c" -> TArray(TStruct())))
  }

  def checkMemo(ir: BaseIR, requestedType: BaseType, expected: Array[BaseType]) {
    val irCopy = ir.deepCopy()
    assert(PruneDeadFields.isSupertype(requestedType, irCopy.typ),
      s"not supertype:\n  super: ${ requestedType.parsableString() }\n  sub:   ${ irCopy.typ.parsableString() }")
    val memo = Memo.empty[BaseType]
    irCopy match {
      case mir: MatrixIR => PruneDeadFields.memoizeMatrixIR(mir, requestedType.asInstanceOf[MatrixType], memo)
      case tir: TableIR => PruneDeadFields.memoizeTableIR(tir, requestedType.asInstanceOf[TableType], memo)
      case ir: IR => PruneDeadFields.memoizeValueIR(ir, requestedType.asInstanceOf[Type], memo)
    }
    irCopy.children.zipWithIndex.foreach { case (child, i) =>
      if (expected(i) != null && expected(i) != memo.lookup(child)) {
        fatal(s"For base IR $ir\n  Child $i\n  Expected: ${ expected(i) }\n  Actual:   ${ memo.lookup(child) }")
      }
    }
  }

  def checkRebuild[T <: BaseIR](
    ir: T,
    requestedType: BaseType,
    f: (T, T) => Boolean = (left: TableIR, right: TableIR) => left == right) {
    val irCopy = ir.deepCopy()
    val memo = Memo.empty[BaseType]
    val rebuilt = (irCopy match {
      case mir: MatrixIR =>
        PruneDeadFields.memoizeMatrixIR(mir, requestedType.asInstanceOf[MatrixType], memo)
        PruneDeadFields.rebuild(mir, memo)
      case tir: TableIR =>
        PruneDeadFields.memoizeTableIR(tir, requestedType.asInstanceOf[TableType], memo)
        PruneDeadFields.rebuild(tir, memo)
      case ir: IR =>
        PruneDeadFields.memoizeValueIR(ir, requestedType.asInstanceOf[Type], memo)
        PruneDeadFields.rebuild(ir, Env.empty[Type], memo)
    }).asInstanceOf[T]
    if (!f(ir, rebuilt))
      fatal(s"IR did not rebuild the same:\n  Base:    $ir\n  Rebuilt: $rebuilt")
  }

  val tab = TableLiteral(Table.parallelize(
    hc,
    FastIndexedSeq(Row("hi", FastIndexedSeq(Row(1)), "bye", Row(2, FastIndexedSeq(Row("bar"))))),
    TStruct("1" -> TString(),
      "2" -> TArray(TStruct("2A" -> TInt32())),
      "3" -> TString(),
      "4" -> TStruct("A" -> TInt32(), "B" -> TArray(TStruct("i" -> TString())))),
    None, None).annotateGlobal(5, TInt32(), "g1").annotateGlobal(10, TInt32(), "g2").value)


  val mat = MatrixLiteral(MatrixType.fromParts(
    TStruct("glob1" -> TInt32(), "glob2" -> TFloat64()),
    FastIndexedSeq("ck"),
    TStruct("ck" -> TString(), "c2" -> TInt32(), "c3" -> TArray(TStruct("cc" -> TInt32()))),
    FastIndexedSeq("rk"),
    FastIndexedSeq("rk"),
    TStruct("rk" -> TInt32(), "r2" -> TStruct("x" -> TInt32()), "r3" -> TArray(TInt32())),
    TStruct("e1" -> TFloat64(), "e2" -> TFloat64())), null)

  val emptyTableDep = TableType(TStruct(), None, TStruct())

  def tableRefBoolean(tt: TableType, fields: String*): IR = {
    var let: IR = True()
    fields.foreach { f =>
      val split = f.split("\\.")
      var ir: IR = split(0) match {
        case "row" => Ref("row", tt.rowType)
        case "global" => Ref("global", tt.globalType)
      }

      split.tail.foreach { field =>
        ir = GetField(ir, field)
      }
      let = Let(genUID(), ir, let)
    }
    let
  }

  def tableRefStruct(tt: TableType, fields: String*): IR = {
    MakeStruct(FastIndexedSeq("foo" -> tableRefBoolean(tt, fields: _*)))
  }

  def matrixRefBoolean(mt: MatrixType, fields: String*): IR = {
    var let: IR = True()
    fields.foreach { f =>
      val split = f.split("\\.")
      var ir: IR = split(0) match {
        case "va" => Ref("va", mt.rvRowType)
        case "sa" => Ref("sa", mt.colType)
        case "g" => Ref("g", mt.entryType)
        case "global" => Ref("global", mt.globalType)
      }

      split.tail.foreach { field =>
        ir = GetField(ir, field)
      }
      let = Let(genUID(), ir, let)
    }
    let
  }

  def matrixRefStruct(mt: MatrixType, fields: String*): IR = {
    MakeStruct(FastIndexedSeq("foo" -> matrixRefBoolean(mt, fields: _*)))
  }

  def subsetType(t: Type, path: Array[String], index: Int): Type = {
    if (index == path.length)
      PruneDeadFields.minimal(t)
    else
      t match {
        case ts: TStruct => TStruct(ts.required, path(index) -> subsetType(ts.field(path(index)).typ, path, index + 1))
        case ta: TArray => TArray(subsetType(ta.elementType, path, index), ta.required)
      }
  }

  def subsetTable(tt: TableType, fields: String*): TableType = {
    val rowFields = new ArrayBuilder[TStruct]()
    val globalFields = new ArrayBuilder[TStruct]()
    fields.foreach { f =>
      val split = f.split("\\.")
      val (head, ab) = split(0) match {
        case "row" => tt.rowType -> rowFields
        case "global" => tt.globalType -> globalFields
      }
      ab += subsetType(head, split, 1).asInstanceOf[TStruct]
    }
    val min = PruneDeadFields.minimal(tt)
    tt.copy(
      rowType = PruneDeadFields.unify(tt.rowType, Array(min.rowType) ++ rowFields.result(): _*),
      globalType = PruneDeadFields.unify(tt.globalType, globalFields.result(): _*)
    )
  }

  def subsetMatrixTable(mt: MatrixType, fields: String*): MatrixType = {
    val rowFields = new ArrayBuilder[TStruct]()
    val colFields = new ArrayBuilder[TStruct]()
    val entryFields = new ArrayBuilder[TStruct]()
    val globalFields = new ArrayBuilder[TStruct]()
    fields.foreach { f =>
      val split = f.split("\\.")
      val (head, ab) = split(0) match {
        case "va" => mt.rvRowType -> rowFields
        case "sa" => mt.colType -> colFields
        case "g" => mt.entryType -> entryFields
        case "global" => mt.globalType -> globalFields
      }
      ab += subsetType(head, split, 1).asInstanceOf[TStruct]
    }
    val min = PruneDeadFields.minimal(mt)
    mt.copyParts(
      globalType = PruneDeadFields.unify(mt.globalType, globalFields.result(): _*),
      colType = PruneDeadFields.unify(mt.colType, Array(min.colType) ++ colFields.result(): _*),
      rowType = PruneDeadFields.unify(mt.rowType, Array(min.rowType) ++ rowFields.result(): _*),
      entryType = PruneDeadFields.unify(mt.entryType, Array(min.entryType) ++ entryFields.result(): _*)
    )
  }

  @Test def testTableJoinMemo() {
    val tk1 = TableKeyBy(tab, Array("1"), 1)
    val tk2 = TableKeyBy(tab, Array("3"), 1)
    val tj = TableJoin(tk1, tk2, "inner")
    checkMemo(tj,
      subsetTable(tj.typ, "row.1", "row.4", "row.1_1"),
      Array(
        subsetTable(tk1.typ, "row.1", "row.4"),
        subsetTable(tk2.typ, "row.1", "row.3")
      )
    )
  }

  @Test def testTableExplodeMemo() {
    val te = TableExplode(tab, "2")
    checkMemo(te, subsetTable(te.typ), Array(subsetTable(tab.typ, "row.2")))
  }

  @Test def testTableFilterMemo() {
    checkMemo(TableFilter(tab, tableRefBoolean(tab.typ, "row.2")),
      subsetTable(tab.typ, "row.3"),
      Array(subsetTable(tab.typ, "row.2", "row.3"), null))
    checkMemo(TableFilter(tab, False()),
      subsetTable(tab.typ, "row.1"),
      Array(subsetTable(tab.typ, "row.1"), TBoolean()))
  }

  @Test def testTableKeyByMemo() {
    val tk = TableKeyBy(tab, Array("1"), 1)
    checkMemo(tk, subsetTable(tk.typ, "row.2"), Array(subsetTable(tab.typ, "row.1", "row.2")))
  }

  @Test def testTableMapRowsEmptyMemo() {
    val tmr = TableMapRows(tab, tableRefStruct(tab.typ, "row.1", "row.2"), None, None)
    checkMemo(tmr, subsetTable(tmr.typ, "row.foo"), Array(subsetTable(tab.typ, "row.1", "row.2"), null))
  }

  @Test def testTableMapGlobalsMemo() {
    val tmg = TableMapGlobals(tab, tableRefStruct(tab.typ, "global.g1"), null)
    checkMemo(tmg, subsetTable(tmg.typ, "global.foo"), Array(subsetTable(tab.typ, "global.g1"), null))
  }

  @Test def testMatrixColsTableMemo() {
    val mct = MatrixColsTable(mat)
    checkMemo(mct,
      subsetTable(mct.typ, "global.glob1", "row.c2"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "sa.c2")))
  }

  @Test def testMatrixRowsTableMemo() {
    val mrt = MatrixRowsTable(mat)
    checkMemo(mrt,
      subsetTable(mrt.typ, "global.glob1", "row.r2"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "va.r2")))
  }

  @Test def testMatrixEntriesTableMemo() {
    val met = MatrixEntriesTable(mat)
    checkMemo(met,
      subsetTable(met.typ, "global.glob1", "row.r2", "row.c2", "row.e2"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "va.r2", "sa.c2", "g.e2")))
  }

  @Test def testTableUnionMemo() {
    checkMemo(
      TableUnion(FastIndexedSeq(tab, tab)),
      subsetTable(tab.typ, "row.1", "global.g1"),
      Array(subsetTable(tab.typ, "row.1", "global.g1"),
        subsetTable(tab.typ, "row.1", "global.g1"))
    )
  }

  @Test def testTableUnkeyMemo() {
    val tk = TableKeyBy(tab, Array("1"), 1)
    val tu = TableUnkey(tk)
    checkMemo(tu, subsetTable(tu.typ, "row.2"),
      Array(subsetTable(tk.typ, "row.2")))
  }

  @Test def testMatrixFilterColsMemo() {
    val mfc = FilterColsIR(mat, matrixRefBoolean(mat.typ, "global.glob1", "sa.c2"))
    checkMemo(mfc,
      subsetMatrixTable(mfc.typ, "sa.c3"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "sa.c2", "sa.c3"), null))
  }

  @Test def testMatrixFilterRowsMemo() {
    val mfr = MatrixFilterRowsIR(mat, matrixRefBoolean(mat.typ, "global.glob1", "va.r2"))
    checkMemo(mfr,
      subsetMatrixTable(mfr.typ, "sa.c3", "va.r3"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "va.r2", "sa.c3", "va.r3"), null))
  }

  @Test def testMatrixFilterEntriesMemo() {
    val mfe = MatrixFilterEntries(mat, matrixRefBoolean(mat.typ, "global.glob1", "va.r2", "sa.c2", "g.e2"))
    checkMemo(mfe,
      subsetMatrixTable(mfe.typ, "sa.c3", "va.r3"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "va.r2", "sa.c3", "sa.c2", "va.r3", "g.e2"), null))
  }

  @Test def testMatrixMapColsMemo() {
    val mmc = MatrixMapCols(mat, matrixRefStruct(mat.typ, "global.glob1", "sa.c2", "va.r2", "g.e2"), Some(FastIndexedSeq()))
    checkMemo(mmc, subsetMatrixTable(mmc.typ, "va.r3", "sa.foo"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "sa.c2", "va.r2", "g.e2", "va.r3"), null))
  }

  @Test def testMatrixMapRowsMemo() {
    val mmr = MatrixMapRows(mat, matrixRefStruct(mat.typ, "global.glob1", "sa.c2", "va.r2", "g.e2"),
      Some(IndexedSeq(), IndexedSeq()))
    checkMemo(mmr, subsetMatrixTable(mmr.typ, "sa.c3", "va.foo"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "sa.c2", "va.r2", "g.e2", "sa.c3"), null))
  }

  @Test def testMatrixMapGlobalsMemo() {
    val mmg = MatrixMapGlobals(mat, matrixRefStruct(mat.typ, "global.glob1"), null)
    checkMemo(mmg, subsetMatrixTable(mmg.typ, "global.foo", "va.r3", "sa.c3"),
      Array(subsetMatrixTable(mat.typ, "global.glob1", "va.r3", "sa.c3"), null))
  }

  @Test def testCollectColsByKeyMemo() {
    val ccbk = CollectColsByKey(mat)
    checkMemo(ccbk,
      subsetMatrixTable(ccbk.typ, "g.e2", "sa.c2"),
      Array(subsetMatrixTable(mat.typ, "g.e2", "sa.c2")))
  }

  @Test def testMatrixAggregateRowsByKeyMemo() {
    val magg = MatrixAggregateRowsByKey(mat,
      matrixRefStruct(mat.typ, "g.e2", "va.r2", "sa.c2"))
    checkMemo(magg,
      subsetMatrixTable(magg.typ, "sa.c3", "g.foo"),
      Array(subsetMatrixTable(mat.typ, "sa.c3", "g.e2", "va.r2", "sa.c2"), null)
    )
  }

  @Test def testMatrixAggregateColsByKeyMemo() {
    val magg = MatrixAggregateColsByKey(mat,
      matrixRefStruct(mat.typ, "g.e2", "va.r2", "sa.c2"))
    checkMemo(magg,
      subsetMatrixTable(magg.typ, "va.r3", "g.foo"),
      Array(subsetMatrixTable(mat.typ, "sa.c2", "va.r2", "va.r3", "g.e2"), null))
  }

  val ref = Ref("x", TStruct("a" -> TInt32(), "b" -> TInt32(), "c" -> TInt32()))
  val arr = MakeArray(FastIndexedSeq(ref, ref), TArray(ref.typ))
  val empty = TStruct()
  val justA = TStruct("a" -> TInt32())
  val justB = TStruct("b" -> TInt32())

  @Test def testIfMemo() {
    checkMemo(If(True(), ref, ref),
      justA,
      Array(TBoolean(), justA, justA))
  }

  @Test def testLetMemo() {
    checkMemo(Let("foo", ref, Ref("foo", ref.typ)), justA, Array(justA, null))
    checkMemo(Let("foo", ref, True()), TBoolean(), Array(empty, null))
  }

  @Test def testMakeArrayMemo() {
    checkMemo(arr, TArray(justB), Array(justB, justB))
  }

  @Test def testArrayRefMemo() {
    checkMemo(ArrayRef(arr, I32(0)), justB, Array(TArray(justB), null))
  }

  @Test def testArrayLenMemo() {
    checkMemo(ArrayLen(arr), TInt32(), Array(TArray(empty)))
  }

  @Test def testArrayMapMemo() {
    checkMemo(ArrayMap(arr, "foo", Ref("foo", ref.typ)),
      TArray(justB), Array(TArray(justB), null))
  }

  @Test def testArrayFilterMemo() {
    checkMemo(ArrayFilter(arr, "foo", Let("foo2", GetField(Ref("foo", ref.typ), "b"), False())),
      TArray(empty), Array(TArray(justB), null))
    checkMemo(ArrayFilter(arr, "foo", False()),
      TArray(empty), Array(TArray(empty), null))
    checkMemo(ArrayFilter(arr, "foo", False()),
      TArray(justB), Array(TArray(justB), null))
  }

  @Test def testArrayFlatMapMemo() {
    checkMemo(ArrayFlatMap(arr, "foo", MakeArray(IndexedSeq(Ref("foo", ref.typ)), TArray(ref.typ))),
      TArray(justA),
      Array(TArray(justA), null))
  }

  @Test def testArrayFoldMemo() {
    checkMemo(ArrayFold(arr, I32(0), "comb", "foo", GetField(Ref("foo", ref.typ), "a")),
      TInt32(),
      Array(TArray(justA), null, null))
  }

  @Test def testArrayForMemo() {
    checkMemo(ArrayFor(arr, "foo", Begin(FastIndexedSeq(GetField(Ref("foo", ref.typ), "a")))),
      TVoid,
      Array(TArray(justA), null))
  }

  @Test def testMakeStructMemo() {
    checkMemo(MakeStruct(Seq("a" -> ref, "b" -> I32(10))),
      TStruct("a" -> justA), Array(justA, null))
    checkMemo(MakeStruct(Seq("a" -> ref, "b" -> I32(10))),
      TStruct(), Array(null, null))
  }

  @Test def testInsertFieldsMemo() {
    checkMemo(InsertFields(ref, Seq("d" -> ref)),
      justA ++ TStruct("d" -> justB),
      Array(justA, justB))
  }

  @Test def testSelectFieldsMemo() {
    checkMemo(SelectFields(ref, Seq("a", "b")), justA, Array(justA))
  }

  @Test def testGetFieldMemo() {
    checkMemo(GetField(ref, "a"), TInt32(), Array(justA))
  }

  @Test def testMakeTupleMemo() {
    checkMemo(MakeTuple(Seq(ref)), TTuple(justA), Array(justA))
  }

  @Test def testGetTupleElementMemo() {
    checkMemo(GetTupleElement(MakeTuple(Seq(ref)), 0), justB, Array(TTuple(justB)))
  }

  @Test def testTableCountMemo() {
    checkMemo(TableCount(tab), TInt64(), Array(subsetTable(tab.typ)))
  }

  @Test def testTableAggregateMemo() {
    checkMemo(TableAggregate(tab, GetField(Ref("global", tab.typ.globalType), "g1")),
      TInt32(),
      Array(subsetTable(tab.typ, "global.g1"), null))
  }

  @Test def testTableImportRebuild() {
    val tt = TableType(
      TStruct("a" -> TInt32(), "b" -> TFloat64()),
      None,
      TStruct())
    val opts = TableReaderOptions(1, Array(), Array(), "", "", true, "", 'a', true, Array(0, 1), originalType = tt.rowType)
    checkRebuild(TableImport(Array(""), tt, opts),
      tt.copy(rowType = TStruct("a" -> TInt32())),
      (_: TableImport, rebuilt: TableImport) =>
        rebuilt.typ == tt.copy(rowType = TStruct("a" -> TInt32())) &&
          rebuilt.readerOpts.useColIndices.sameElements(Array(0)))
  }

  @Test def testTableFilterRebuild() {

  }

  @Test def testTableMapRowsRebuild() {

  }

  @Test def testTableMapGlobalsRebuild() {

  }

  @Test def testMatrixFilterColsRebuild() {

  }

  @Test def testMatrixFilterEntriesRebuild() {

  }

  @Test def testMatrixMapRowsRebuild() {

  }

  @Test def testMatrixMapColsRebuild() {

  }

  @Test def testMatrixMapEntriesRebuild() {

  }

  @Test def testMatrixMapGlobalsRebuild() {

  }

  @Test def testMatrixAggregateRowsByKeyRebuild() {

  }

  @Test def testNARebuild() {

  }

  @Test def testIfRebuild() {

  }

  @Test def testLetRebuild() {

  }

  @Test def testRefRebuild() {

  }

  @Test def testMakeArrayRebuild() {

  }

  @Test def testArrayMapRebuild() {

  }

  @Test def testArrayFilterRebuild() {

  }

  @Test def testArrayFlatmapRebuild() {

  }

  @Test def testArrayFoldRebuild() {

  }

  @Test def testArrayForRebuild() {

  }

  @Test def testMakeStructRebuild() {

  }

  @Test def testInsertFieldsRebuild() {

  }

  @Test def testSelectFieldsRebuild() {

  }

  @Test def testTableAggregateRebuild() {

  }
}

