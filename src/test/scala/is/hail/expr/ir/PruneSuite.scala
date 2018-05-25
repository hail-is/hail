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
          "aaa" -> TString()
        )
      ),
      "b" -> TInt32(),
      "c" -> TArray(TStruct(
        "ca" -> TInt32()
      ))
    )

    assert(PruneDeadFields.unify(base, TStruct()) == TStruct())
    assert(PruneDeadFields.unify(base, TStruct("b" -> TInt32())) == TStruct("b" -> TInt32()))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct())) == TStruct("a" -> TStruct()))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct()),
      TStruct("b" -> TInt32())) == TStruct("a" -> TStruct(), "b" -> TInt32()))
    assert(PruneDeadFields.unify(base, TStruct("c" -> TArray(TStruct()))) == TStruct("c" -> TArray(TStruct())))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct("ab" -> TStruct())),
      TStruct("c" -> TArray(TStruct()))) == TStruct("a" -> TStruct("ab" -> TStruct()), "c" -> TArray(TStruct())))
  }

  @Test def testPipeline() {
    hc // initialize
    var ir = TableParallelize(TableType(TStruct(
      "a" -> TString(),
      "b" -> TString(),
      "c" -> TStruct("c1" -> TInt64(), "c2" -> TArray(TInt32())),
      "d" -> TArray(TStruct("d1" -> TInt32(), "d2" -> TFloat64()))),
      Some(Array("a")),
      TStruct()),
      IndexedSeq(
        Row("foo", "row1-b", Row(1L, IndexedSeq(1, 2, 3)), IndexedSeq(Row(1, 1.5), Row(2, 2.5))),
        Row("bar", "row2-b", Row(10L, IndexedSeq(-1, -2, -3)), IndexedSeq(Row(5, 5.5), Row(7, 7.5)))))

    val sig = AggSignature(Sum(), TInt64(), Seq(), None)
    var ir2 = TableFilter(ir, ApplyComparisonOp(LT(TInt32()), GetField(ArrayRef(GetField(Ref("row", ir.typ.rowType), "d"), I32(0)), "d1"), I32(3)))
    var aggir2 = TableAggregate(ir2, ApplyAggOp(SeqOp(GetField(GetField(Ref("row", ir2.typ.rowType), "c"), "c1"), I32(0), sig), FastIndexedSeq(), None, sig))

    val row = Ref("row", ir.typ.rowType)
    var ir3 = TableMapRows(ir, MakeStruct(Seq(
      "a" -> GetField(row, "a"),
      "b" -> GetField(row, "b"),
      "c" -> GetField(row, "c"),
      "d" -> GetField(row, "d"),
      "e" -> I32(10))), None, None)

    var aggir3 = TableAggregate(ir3, ApplyAggOp(SeqOp(GetField(GetField(Ref("row", ir3.typ.rowType), "c"), "c1"), I32(0), sig), FastIndexedSeq(), None, sig))
    assert(Interpret[Long](aggir3, optimize = true) == Interpret[Long](aggir3, optimize = false))
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
      val memoDep = memo.lookup(child)
      if (expected(i) != null && expected(i) != memoDep) {
        fatal(s"For base IR $ir\n  Child $i\n  Expected: ${ expected(i) }\n  Actual:   $memoDep")
      }
    }
  }

  def checkRebuild(ir: BaseIR, requestedType: BaseType, expected: BaseIR) {
    val irCopy = ir.deepCopy()
    val memo = Memo.empty[BaseType]
    val rebuilt = irCopy match {
      case mir: MatrixIR =>
        PruneDeadFields.memoizeMatrixIR(mir, requestedType.asInstanceOf[MatrixType], memo)
        PruneDeadFields.rebuild(mir, memo)
      case tir: TableIR =>
        PruneDeadFields.memoizeTableIR(tir, requestedType.asInstanceOf[TableType], memo)
        PruneDeadFields.rebuild(tir, memo)
      case ir: IR =>
        PruneDeadFields.memoizeValueIR(ir, requestedType.asInstanceOf[Type], memo)
        PruneDeadFields.rebuild(ir, Env.empty[Type], memo)
    }
    if (rebuilt != expected)
      fatal(s"IR did not rebuild the same:\n  Expect: $expected\n  Actual: $rebuilt")
  }

  val tableLiteral = TableLiteral(Table.parallelize(
    hc,
    FastIndexedSeq(Row("hi", FastIndexedSeq(Row(1)), "bye", Row(2, FastIndexedSeq(Row("bar"))))),
    TStruct("1" -> TString(),
      "2" -> TArray(TStruct("2A" -> TInt32())),
      "3" -> TString(),
      "4" -> TStruct("A" -> TInt32(), "B" -> TArray(TStruct("i" -> TString())))),
    None, None).annotateGlobal(5, TInt32(), "x").value)


  val matrixLiteral = MatrixLiteral(MatrixType.fromParts(
    TStruct("glob1" -> TInt32()),
    FastIndexedSeq("ck"),
    TStruct("ck" -> TString(), "c2" -> TInt32(), "c3" -> TArray(TStruct("cc" -> TInt32()))),
    FastIndexedSeq("rk"),
    FastIndexedSeq("rk"),
    TStruct("rk" -> TInt32(), "r2" -> TStruct("x" -> TInt32()), "r3" -> TArray(TInt32())),
    TStruct("entry1" -> TFloat64(), "entry2" -> TFloat64())), null)

  val emptyTableDep = TableType(TStruct(), None, TStruct())

  @Test def testTableJoinMemo() {
    val tj = TableJoin(TableKeyBy(tableLiteral, Array("1"), 1), TableKeyBy(tableLiteral, Array("3"), 1), "inner")
    val dep = TableType(
      globalType = TStruct(),
      key = Some(FastIndexedSeq("1")),
      rowType = TStruct("1" -> TString(), "4" -> TStruct("A" -> TInt32()), "1_1" -> TString()))
    checkMemo(tj, dep,
      Array(
        TableType(
          globalType = TStruct(),
          key = Some(FastIndexedSeq("1")),
          rowType = TStruct("1" -> TString(), "4" -> TStruct("A" -> TInt32()))),
        TableType(
          globalType = TStruct(),
          key = Some(FastIndexedSeq("3")),
          rowType = TStruct("1" -> TString(), "3" -> TString()))
      )
    )
  }

  @Test def testTableExplodeMemo() {
    val te = TableExplode(tableLiteral, "2")
    checkMemo(te, tableLiteral.typ.copy(rowType = TStruct()), Array(tableLiteral.typ.copy(rowType = TStruct("2" -> TArray(TStruct())))))
  }

  @Test def testTableFilterMemo() {
    val row = Ref("row", tableLiteral.typ.rowType)
    val tf = TableFilter(tableLiteral, ApplyComparisonOp(GT(TInt32()), ArrayLen(GetField(row, "2")), GetField(GetField(row, "4"), "A")))
    checkMemo(tf, emptyTableDep,
      Array(tableLiteral.typ.copy(
        globalType = TStruct(),
        rowType = TStruct("2" -> TArray(TStruct()), "4" -> TStruct("A" -> TInt32()))),
        TBoolean()))
    checkMemo(TableFilter(tableLiteral, False()), emptyTableDep, Array(emptyTableDep, TBoolean()))
  }

  @Test def testTableKeyByMemo() {
    val tk = TableKeyBy(tableLiteral, Array("1"), 1)
    checkMemo(tk, TableType(TStruct("1" -> TString()), Some(FastIndexedSeq("1")), TStruct()),
      Array(TableType(TStruct("1" -> TString()), None, TStruct())))
  }

  @Test def testTableMapRowsEmptyMemo() {
    val tmr = TableMapRows(tableLiteral, MakeStruct(FastIndexedSeq()), None, None)
    checkMemo(tmr, emptyTableDep, Array(emptyTableDep, null))
  }

  @Test def testTableMapRowsSelectMemo() {
    val row = Ref("row", tableLiteral.typ.rowType)
    val tmr = TableMapRows(tableLiteral, MakeStruct(FastSeq("foo" -> GetField(row, "1"))), None, None)
    checkMemo(tmr, emptyTableDep.copy(rowType = TStruct("foo" -> TString())),
      Array(emptyTableDep.copy(rowType = TStruct("1" -> TString())), null))
  }

  @Test def testTableMapGlobalsMemo() {
    val tmg = TableMapGlobals(tableLiteral, MakeStruct(FastIndexedSeq()), null)
    checkMemo(tmg, tableLiteral.typ.copy(globalType = TStruct()), Array(tableLiteral.typ.copy(globalType = TStruct()), null))
  }

  @Test def testMatrixColsTableMemo() {
    checkMemo(MatrixColsTable(matrixLiteral),
      TableType(TStruct("ck" -> TString()), Some(FastIndexedSeq("ck")), TStruct()),
      Array(matrixLiteral.typ.copyParts(
        rowType = TStruct("rk" -> TInt32()),
        globalType = TStruct(),
        colType = TStruct("ck" -> TString()),
        entryType = TStruct())))
  }

  @Test def testMatrixRowsTableMemo() {
    checkMemo(MatrixRowsTable(matrixLiteral),
      TableType(TStruct("rk" -> TInt32()), Some(FastIndexedSeq("rk")), TStruct()),
      Array(matrixLiteral.typ.copyParts(
        rowType = TStruct("rk" -> TInt32()),
        globalType = TStruct(),
        colType = TStruct("ck" -> TString()),
        entryType = TStruct())))
  }

  @Test def testMatrixEntriesTableMemo() {
    checkMemo(MatrixEntriesTable(matrixLiteral),
      TableType(TStruct("rk" -> TInt32(), "ck" -> TString(), "c3" -> TArray(TStruct()), "entry1" -> TFloat64()),
        Some(FastIndexedSeq("rk", "ck")), TStruct()),
      Array(matrixLiteral.typ.copyParts(
        rowType = TStruct("rk" -> TInt32()),
        globalType = TStruct(),
        colType = TStruct("ck" -> TString(), "c3" -> TArray(TStruct())),
        entryType = TStruct("entry1" -> TFloat64()))))
  }

  @Test def testMatrixEntriesTableNoEntryFieldsMemo() {
    checkMemo(MatrixEntriesTable(matrixLiteral),
      TableType(TStruct("rk" -> TInt32(), "ck" -> TString(), "c3" -> TArray(TStruct())),
        Some(FastIndexedSeq("rk", "ck")), TStruct()),
      Array(matrixLiteral.typ.copyParts(
        rowType = TStruct("rk" -> TInt32()),
        globalType = TStruct(),
        colType = TStruct("ck" -> TString(), "c3" -> TArray(TStruct())),
        entryType = TStruct())))
  }

  @Test def testTableUnionMemo() {
    val tu = TableUnion(FastIndexedSeq(tableLiteral, tableLiteral))
    val dep = tableLiteral.typ.copy(rowType = TStruct("2" -> TArray(TStruct())))
    checkMemo(tu, dep, Array(dep, dep))
  }

  @Test def testTableUnkeyMemo() {
    val tu = TableUnkey(TableKeyBy(tableLiteral, Array("1"), 1))
    checkMemo(tu, emptyTableDep.copy(rowType = TStruct("2" -> TArray(TStruct()))),
      Array(TableType(TStruct("1" -> TString(), "2" -> TArray(TStruct())), Some(FastIndexedSeq("1")), TStruct())))
  }

  @Test def testMatrixFilterColsMemo() {
    val fc1 = FilterColsIR(matrixLiteral, True())
    val fc2 = FilterColsIR(matrixLiteral,
      ApplyComparisonOp(GT(TInt32()), GetField(Ref("global", matrixLiteral.typ.globalType), "glob1"), I32(0)))
    val dep1 = matrixLiteral.typ.copyParts(colType = TStruct("ck" -> TString()), entryType = TStruct())
    checkMemo(fc1, dep1, Array(dep1, null))
    checkMemo(fc2, dep1, Array(dep1.copy(globalType = TStruct("glob1" -> TInt32())), null))
  }

  @Test def testMatrixFilterRowsMemo() {
    val fr1 = MatrixFilterRowsIR(matrixLiteral, True())
    val fr2 = MatrixFilterRowsIR(matrixLiteral,
      ApplyComparisonOp(GT(TInt32()), GetField(Ref("global", matrixLiteral.typ.globalType), "glob1"), I32(0)))
    val dep1 = matrixLiteral.typ.copyParts(colType = TStruct("ck" -> TString()), entryType = TStruct())
    checkMemo(fr1, dep1, Array(dep1, null))
    checkMemo(fr2, dep1, Array(dep1.copy(globalType = TStruct("glob1" -> TInt32())), null))
  }

  @Test def testMatrixFilterEntriesMemo() {
    val fe1 = MatrixFilterEntries(matrixLiteral, True())
    val fe2 = MatrixFilterEntries(matrixLiteral,
      ApplyComparisonOp(GT(TInt32()), GetField(Ref("global", matrixLiteral.typ.globalType), "glob1"), I32(0)))
    val dep1 = matrixLiteral.typ.copyParts(colType = TStruct("ck" -> TString()), entryType = TStruct())
    checkMemo(fe1, dep1, Array(dep1, null))
    checkMemo(fe2, dep1, Array(dep1.copy(globalType = TStruct("glob1" -> TInt32())), null))
  }

  @Test def testMatrixMapEntriesMemo() {
    val mme = MatrixMapEntries(matrixLiteral, MakeStruct(FastIndexedSeq("x" -> ApplyBinaryPrimOp(Add(),
      GetField(Ref("global", matrixLiteral.typ.globalType), "glob1"),
      GetField(Ref("sa", matrixLiteral.typ.colType), "c2")))))
    checkMemo(mme, matrixLiteral.typ.copyParts(rowType = TStruct("rk" -> TInt32()),
      colType = TStruct("ck" -> TString()),
      entryType = TStruct("x" -> TInt32())), Array(matrixLiteral.typ.copyParts(rowType = TStruct("rk" -> TInt32()),
      colType = TStruct("ck" -> TString(), "c2" -> TInt32()),
      entryType = TStruct()), null))
  }

  @Test def testMatrixMapColsMemo() {
    val mmc = MatrixMapCols(matrixLiteral, MakeStruct(FastIndexedSeq(
      "ck" -> GetField(Ref("sa", matrixLiteral.typ.colType), "ck"),
      "x" -> ApplyBinaryPrimOp(Add(),
        GetField(Ref("global", matrixLiteral.typ.globalType), "glob1"),
        GetField(Ref("sa", matrixLiteral.typ.colType), "c2")))), None)
    checkMemo(mmc, matrixLiteral.typ.copyParts(rowType = TStruct("rk" -> TInt32()),
      entryType = TStruct(),
      colType = TStruct("ck" -> TString(), "x" -> TInt32())),
      Array(matrixLiteral.typ.copyParts(rowType = TStruct("rk" -> TInt32()),
        colType = TStruct("ck" -> TString(), "c2" -> TInt32()),
        entryType = TStruct()), null))
  }

  @Test def testMatrixMapRowsMemo() {
    val mmr = MatrixMapRows(matrixLiteral, MakeStruct(FastIndexedSeq(
      "rk" -> GetField(Ref("va", matrixLiteral.typ.rvRowType), "rk"),
      "x" -> ApplyBinaryPrimOp(Add(),
        GetField(Ref("global", matrixLiteral.typ.globalType), "glob1"),
        GetField(GetField(Ref("va", matrixLiteral.typ.rvRowType), "r2"), "x")))), None)
    checkMemo(mmr,
      matrixLiteral.typ.copyParts(rowType = TStruct("rk" -> TInt32(), "x" -> TInt32()),
        entryType = TStruct(),
        colType = TStruct("ck" -> TString())),
      Array(matrixLiteral.typ.copyParts(rowType = TStruct("rk" -> TInt32(), "r2" -> TStruct("x" -> TInt32())),
        colType = TStruct("ck" -> TString()),
        entryType = TStruct()), null))
  }

  @Test def testMatrixMapGlobalsMemo() {
    val mmg = MatrixMapGlobals(matrixLiteral, MakeStruct(FastIndexedSeq()), null)
    checkMemo(mmg, matrixLiteral.typ.copy(globalType = TStruct()),
      Array(matrixLiteral.typ.copy(globalType = TStruct()), null))
  }

  @Test def testCollectColsByKeyMemo() {
    checkMemo(CollectColsByKey(matrixLiteral),
      MatrixType.fromParts(
        TStruct(),
        FastIndexedSeq("ck"),
        TStruct("ck" -> TString(), "c2" -> +TArray(TInt32())),
        FastIndexedSeq("rk"),
        FastIndexedSeq("rk"),
        TStruct("rk" -> TInt32(), "r2" -> TStruct("x" -> TInt32())),
        TStruct("entry2" -> +TArray(TFloat64()))),
      Array(MatrixType.fromParts(
        TStruct(),
        FastIndexedSeq("ck"),
        TStruct("ck" -> TString(), "c2" -> TInt32()),
        FastIndexedSeq("rk"),
        FastIndexedSeq("rk"),
        TStruct("rk" -> TInt32(), "r2" -> TStruct("x" -> TInt32())),
        TStruct("entry2" -> TFloat64())), null))
  }

  @Test def testMatrixAggregateRowsByKeyMemo() {
    val aggSig = AggSignature(Sum(), TFloat64(), Seq(), None)
    val magg = MatrixAggregateRowsByKey(matrixLiteral,
      MakeStruct(FastIndexedSeq("foo" -> ApplyAggOp(SeqOp(GetField(Ref("g", matrixLiteral.typ.entryType), "entry1"), I32(0), aggSig),
        FastIndexedSeq(), None, aggSig))))
    checkMemo(magg, MatrixType.fromParts(
      TStruct(),
      FastIndexedSeq("ck"),
      TStruct("ck" -> TString()),
      FastIndexedSeq("rk"),
      FastIndexedSeq("rk"),
      TStruct("rk" -> TInt32()),
      TStruct("foo" -> TFloat64())),
      Array(
        MatrixType.fromParts(
          TStruct(),
          FastIndexedSeq("ck"),
          TStruct("ck" -> TString()),
          FastIndexedSeq("rk"),
          FastIndexedSeq("rk"),
          TStruct("rk" -> TInt32()),
          TStruct("entry1" -> TFloat64())),
        null
      )
    )
  }

}

