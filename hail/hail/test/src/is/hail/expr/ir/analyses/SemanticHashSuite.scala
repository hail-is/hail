package is.hail.expr.ir.analyses

import is.hail.{HailSuite, PrettyVersion}
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.io.fs.{FS, FakeFS, FakeURL, FileListEntry}
import is.hail.linalg.BlockMatrixMetadata
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.virtual._

import java.io.FileNotFoundException
import java.lang

import org.json4s.JValue

class SemanticHashSuite extends HailSuite {

  private[this] val fakeFs: FS =
    new FakeFS {
      override def eTag(url: FakeURL): Option[String] =
        Some(url.path)

      override def glob(url: FakeURL): IndexedSeq[FileListEntry] =
        ArraySeq(new FileListEntry {
          override def getPath: String = url.path
          override def getActualUrl: String = url.path
          override def getModificationTime: lang.Long = ???
          override def getLen: Long = ???
          override def isDirectory: Boolean = ???
          override def isSymlink: Boolean = ???
          override def isFile: Boolean = true
          override def getOwner: String = ???
        })
    }

  def mkRelationalLet(bindings: IndexedSeq[(Name, IR)], body: IR): IR =
    bindings.foldRight(body) { case ((name, value), body) =>
      RelationalLet(name, value, body)
    }

  object checkSemanticEquivalence extends TestCases {
    def apply(
      a: BaseIR,
      b: BaseIR,
      isEqual: Boolean,
      comment: String,
    )(implicit loc: munit.Location
    ): Unit = test("SemanticEquivalence") {
      ctx.local(fs = fakeFs) { ctx =>
        assertEquals(
          SemanticHash(ctx, a) == SemanticHash(ctx, b),
          isEqual,
          s"expected semhash($a) ${if (isEqual) "==" else "!="} semhash($b), $comment",
        )
      }
    }
  }

  // trivial value semantics
  checkSemanticEquivalence(True(), True(), true, "Refl")
  checkSemanticEquivalence(False(), False(), true, "Refl")
  checkSemanticEquivalence(True(), False(), false, "Refl")
  checkSemanticEquivalence(I32(0), I32(0), true, "Refl")
  checkSemanticEquivalence(I32(0), I32(1), false, "Refl")
  checkSemanticEquivalence(I64(0), I64(0), true, "Refl")
  checkSemanticEquivalence(I64(0), I64(1), false, "Refl")
  checkSemanticEquivalence(F32(0), F32(0), true, "Refl")
  checkSemanticEquivalence(F32(0), F32(1), false, "Refl")
  checkSemanticEquivalence(Void(), Void(), true, "Refl")
  checkSemanticEquivalence(Str("a"), Str("a"), true, "Refl")
  checkSemanticEquivalence(Str("a"), Str("b"), false, "Refl")
  checkSemanticEquivalence(NA(TInt32), NA(TInt32), true, "Refl")
  checkSemanticEquivalence(NA(TInt32), NA(TFloat64), false, "Refl")

  // let-binding semantics
  {
    val x = freshName()
    val y = freshName()

    def letCases(let: (IndexedSeq[(Name, IR)], IR) => IR, ref: (Name, Type) => IR): Unit = {
      checkSemanticEquivalence(
        let(FastSeq(x -> I32(0)), ref(x, TInt32)),
        let(FastSeq(y -> I32(0)), ref(y, TInt32)),
        true,
        "names used in let-bindings do not change semantics",
      )
      checkSemanticEquivalence(
        let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
        let(FastSeq(y -> I32(0), x -> I32(0)), ref(y, TInt32)),
        true,
        "names of let-bindings do not change semantics",
      )
      checkSemanticEquivalence(
        let(FastSeq(x -> I32(0)), ref(x, TInt32)),
        let(FastSeq(x -> I64(0)), ref(x, TInt64)),
        false,
        "different IRs",
      )
      checkSemanticEquivalence(
        let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
        let(FastSeq(y -> I32(0), x -> I32(0)), ref(x, TInt32)),
        false,
        "Different binding being referenced",
      )
      /* `SemanticHash` does not perform or recognise opportunities for simplification.
       * The following examples demonstrate some of its limitations as a consequence. */
      checkSemanticEquivalence(
        let(FastSeq(x -> I32(0)), ref(x, TInt32)),
        let(FastSeq(x -> let(FastSeq(freshName() -> I32(0)), I32(0))), ref(x, TInt32)),
        false,
        "SemanticHash does not simplify",
      )
      checkSemanticEquivalence(
        let(FastSeq(x -> I32(0)), ref(x, TInt32)),
        let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
        false,
        "SemanticHash does not simplify",
      )
    }

    letCases(Let(_, _), Ref)
    letCases(mkRelationalLet _, RelationalRef)
  }

  // struct/tuple semantics
  checkSemanticEquivalence(
    MakeStruct(ArraySeq.empty),
    MakeStruct(ArraySeq.empty),
    true,
    "empty structs",
  )

  checkSemanticEquivalence(
    MakeStruct(ArraySeq(genUID() -> I32(0))),
    MakeStruct(ArraySeq(genUID() -> I32(0))),
    true,
    "field names do not affect MakeStruct semantics",
  )

  checkSemanticEquivalence(
    MakeTuple(ArraySeq.empty),
    MakeTuple(ArraySeq.empty),
    true,
    "empty tuples",
  )

  checkSemanticEquivalence(
    MakeTuple(ArraySeq(0 -> I32(0))),
    MakeTuple(ArraySeq(0 -> I32(0))),
    true,
    "identical tuples",
  )

  checkSemanticEquivalence(
    MakeTuple(ArraySeq(0 -> I32(0))),
    MakeTuple(ArraySeq(1 -> I32(0))),
    false,
    "tuple indices affect MakeTuple semantics",
  )

  checkSemanticEquivalence(
    bindIR(NA(TStruct("0" -> TInt32)))(GetField(_, "0")),
    bindIR(NA(TStruct("1" -> TInt32)))(GetField(_, "1")),
    true,
    "field names do not affect GetField semantics",
  )

  checkSemanticEquivalence(
    bindIR(NA(TTuple(TInt32)))(GetTupleElement(_, 0)),
    bindIR(NA(TTuple(TInt32)))(GetTupleElement(_, 0)),
    true,
    "GetTupleElement of same index",
  )

  checkSemanticEquivalence(
    bindIR(NA(TTuple(ArraySeq(TupleField(0, TInt32)))))(GetTupleElement(_, 0)),
    bindIR(NA(TTuple(ArraySeq(TupleField(1, TInt32)))))(GetTupleElement(_, 1)),
    false,
    "GetTupleElement on different index",
  )

  // tree structure semantics
  checkSemanticEquivalence(
    MakeArray(MakeArray(I32(0)), MakeArray(I32(0))),
    MakeArray(MakeArray(MakeArray(I32(0), I32(0)))),
    false,
    "Tree structure contributes to semantics",
  )

  // table IR semantics
  {
    val ttype = TableType(TStruct("a" -> TInt32, "b" -> TStruct()), IndexedSeq("a"), TStruct())
    val ttypeb = TableType(TStruct("c" -> TInt32, "d" -> TStruct()), IndexedSeq(), TStruct())

    def mkTableRead(reader: TableReader): TableIR =
      TableRead(typ = reader.fullType, dropRows = false, tr = reader)

    def mkTableIR(ttype: TableType, path: String): TableIR =
      mkTableRead(new TableNativeReader(
        TableNativeReaderParameters(path, None),
        mkFakeTableSpec(ttype),
      ))

    val tir = mkTableIR(ttype, "/fake/table")

    checkSemanticEquivalence(tir, tir, true, "TableRead same table")
    checkSemanticEquivalence(
      tir,
      mkTableIR(ttype, "/another/fake/table"),
      false,
      "TableRead different table",
    )
    checkSemanticEquivalence(
      TableKeyBy(tir, IndexedSeq("a")),
      TableKeyBy(tir, IndexedSeq("a")),
      true,
      "TableKeyBy same key",
    )
    checkSemanticEquivalence(
      TableKeyBy(tir, IndexedSeq("a")),
      TableKeyBy(tir, IndexedSeq("b")),
      false,
      "TableKeyBy different key",
    )

    Array[String => TableIR](
      path =>
        mkTableRead(new StringTableReader(
          StringTableReaderParameters(ArraySeq(path), None, false, false, false),
          fakeFs.glob(path),
        )),
      path =>
        mkTableRead(TableNativeZippedReader(
          path + ".left",
          path + ".right",
          None,
          mkFakeTableSpec(ttype),
          mkFakeTableSpec(ttypeb),
        )),
    ).foreach { reader =>
      checkSemanticEquivalence(
        reader("/fake/table"),
        reader("/fake/table"),
        true,
        "read same table",
      )
      checkSemanticEquivalence(
        reader("/fake/table"),
        reader("/another/fake/table"),
        false,
        "read different table",
      )
    }

    Array[TableIR => BaseIR](
      TableGetGlobals,
      TableAggregate(_, I32(0)),
      TableAggregateByKey(_, MakeStruct(FastSeq())),
      TableKeyByAndAggregate(
        _,
        MakeStruct(FastSeq()),
        MakeStruct(FastSeq("idx" -> I32(0))),
        None,
        256,
      ),
      (ir: TableIR) => TableCollect(TableKeyBy(ir, FastSeq())),
      TableCount,
      TableDistinct,
      TableFilter(_, True()),
      TableMapGlobals(_, MakeStruct(IndexedSeq.empty)),
      TableMapRows(_, MakeStruct(FastSeq("a" -> I32(0)))),
      TableRename(_, Map.empty, Map.empty),
    ).foreach(wrap => checkSemanticEquivalence(wrap(tir), wrap(tir), true, ""))
  }

  // block matrix IR semantics
  {
    Array[String => BlockMatrixIR](
      path => BlockMatrixRead(BlockMatrixBinaryReader(path, ArraySeq(1L, 1L), 1)),
      path =>
        BlockMatrixRead(new BlockMatrixNativeReader(
          BlockMatrixNativeReaderParameters(path),
          BlockMatrixMetadata(1, 1, 1, None, IndexedSeq.empty),
        )),
    ).foreach { reader =>
      checkSemanticEquivalence(
        reader("/fake/block-matrix"),
        reader("/fake/block-matrix"),
        true,
        "Read same block matrix",
      )
      checkSemanticEquivalence(
        reader("/fake/block-matrix"),
        reader("/another/fake/block-matrix"),
        false,
        "Read different block matrix",
      )
    }
  }

  test("FileNotFoundExceptions") {
    val fs =
      new FakeFS {
        override def eTag(url: FakeURL): Option[String] =
          throw new FileNotFoundException(url.path)
      }

    val ir = importMatrix("gs://fake-bucket/fake-matrix")

    ctx.local(fs = fs) { ctx =>
      assertEquals(
        SemanticHash(ctx, ir),
        None,
        "SemHash should be resilient to FileNotFoundExceptions.",
      )
    }
  }

  def importMatrix(path: String): MatrixIR = {
    val ty =
      MatrixType(
        TStruct.empty,
        FastSeq("col_idx"),
        TStruct("col_idx" -> TInt32),
        FastSeq("row_idx"),
        TStruct("row_idx" -> TInt32),
        TStruct.empty,
      )

    val reader =
      new MatrixNativeReader(
        MatrixNativeReaderParameters(path, None),
        new AbstractMatrixTableSpec {
          override def matrix_type: MatrixType =
            ty
          override def references_rel_path: String =
            "references"
          override def globalsSpec: AbstractTableSpec =
            mkFakeTableSpec(ty.canonicalTableType)
          override def colsSpec: AbstractTableSpec =
            mkFakeTableSpec(ty.colsTableType)
          override def rowsSpec: AbstractTableSpec =
            mkFakeTableSpec(ty.rowsTableType)
          override def entriesSpec: AbstractTableSpec =
            mkFakeTableSpec(ty.entriesTableType)
          override def file_version: Int =
            1
          override def hail_version: String =
            PrettyVersion
          override def components: Map[String, ComponentSpec] =
            Map.empty
          override def toJValue: JValue = ???
        },
      )

    MatrixRead(ty, false, false, reader)
  }

  def mkFakeTableSpec(ttype: TableType): AbstractTableSpec =
    new AbstractTableSpec {
      override def references_rel_path: String = ???

      override def table_type: TableType = ttype

      override def rowsSpec: AbstractRVDSpec = ???

      override def globalsSpec: AbstractRVDSpec = ???

      override def file_version: Int = 0

      override def hail_version: String = ???

      override def components: Map[String, ComponentSpec] =
        Map("partition_counts" -> PartitionCountsComponentSpec(ArraySeq(1L)))

      override def toJValue: JValue = ???
    }
}
