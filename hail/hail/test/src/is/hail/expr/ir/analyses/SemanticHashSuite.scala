package is.hail.expr.ir.analyses

import is.hail.{ParameterizedTest, PrettyVersion}
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.io.fs.{FakeURL, FileListEntry}
import is.hail.linalg.BlockMatrixMetadata
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.virtual._

import java.io.FileNotFoundException
import java.lang

import org.json4s.JValue
import org.junit.jupiter.api.Test

class SemanticHashSuite {

  def isTriviallySemanticallyEquivalent: ArraySeq[(BaseIR, BaseIR, Boolean, String)] =
    ArraySeq(
      (True(), True(), true, "Refl"),
      (False(), False(), true, "Refl"),
      (True(), False(), false, "Refl"),
      (I32(0), I32(0), true, "Refl"),
      (I32(0), I32(1), false, "Refl"),
      (I64(0), I64(0), true, "Refl"),
      (I64(0), I64(1), false, "Refl"),
      (F32(0), F32(0), true, "Refl"),
      (F32(0), F32(1), false, "Refl"),
      (Void(), Void(), true, "Refl"),
      (Str("a"), Str("a"), true, "Refl"),
      (Str("a"), Str("b"), false, "Refl"),
      (NA(TInt32), NA(TInt32), true, "Refl"),
      (NA(TInt32), NA(TFloat64), false, "Refl"),
    )

  def mkRelationalLet(bindings: IndexedSeq[(Name, IR)], body: IR): IR =
    bindings.foldRight(body) { case ((name, value), body) =>
      RelationalLet(name, value, body)
    }

  def isLetSemanticallyEquivalent = {
    val x = freshName()
    val y = freshName()
    ArraySeq((Let(_, _), Ref), (mkRelationalLet _, RelationalRef)).flatMap { case (let, ref) =>
      ArraySeq[(BaseIR, BaseIR, Boolean, String)](
        (
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(y -> I32(0)), ref(y, TInt32)),
          true,
          "names used in let-bindings do not change semantics",
        ),
        (
          let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
          let(FastSeq(y -> I32(0), x -> I32(0)), ref(y, TInt32)),
          true,
          "names of let-bindings do not change semantics",
        ),
        (
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(x -> I64(0)), ref(x, TInt64)),
          false,
          "different IRs",
        ),
        (
          let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
          let(FastSeq(y -> I32(0), x -> I32(0)), ref(x, TInt32)),
          false,
          "Different binding being referenced",
        ),
        /* `SemanticHash` does not perform or recognise opportunities for simplification.
         * The following examples demonstrate some of its limitations as a consequence. */
        (
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(x -> let(FastSeq(freshName() -> I32(0)), I32(0))), ref(x, TInt32)),
          false,
          "SemanticHash does not simplify",
        ),
        (
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
          false,
          "SemanticHash does not simplify",
        ),
      )
    }
  }

  def isBaseStructSemanticallyEquivalent: ArraySeq[(BaseIR, BaseIR, Boolean, String)] = {

    def f(mkType: Int => Type, get: (IR, Int) => IR, isSame: Boolean, reason: String) = {
      val irs = Array.tabulate(2)(idx => bindIR(NA(mkType(idx)))(get(_, idx)))
      (irs(0): BaseIR, irs(1): BaseIR, isSame, reason)
    }

    ArraySeq[(BaseIR, BaseIR, Boolean, String)](
      (
        MakeStruct(ArraySeq.empty),
        MakeStruct(ArraySeq.empty),
        true,
        "empty structs",
      ),
      (
        MakeStruct(ArraySeq(genUID() -> I32(0))),
        MakeStruct(ArraySeq(genUID() -> I32(0))),
        true,
        "field names do not affect MakeStruct semantics",
      ),
      (
        MakeTuple(ArraySeq.empty),
        MakeTuple(ArraySeq.empty),
        true,
        "empty tuples",
      ),
      (
        MakeTuple(ArraySeq(0 -> I32(0))),
        MakeTuple(ArraySeq(0 -> I32(0))),
        true,
        "identical tuples",
      ),
      (
        MakeTuple(ArraySeq(0 -> I32(0))),
        MakeTuple(ArraySeq(1 -> I32(0))),
        false,
        "tuple indices affect MakeTuple semantics",
      ),
      f(
        mkType = i => TStruct(i.toString -> TInt32),
        get = (ir, i) => GetField(ir, i.toString),
        isSame = true,
        "field names do not affect GetField semantics",
      ),
      f(
        mkType = _ => TTuple(TInt32),
        get = (ir, _) => GetTupleElement(ir, 0),
        isSame = true,
        "GetTupleElement of same index",
      ),
      f(
        mkType = i => TTuple(ArraySeq(TupleField(i, TInt32))),
        get = (ir, i) => GetTupleElement(ir, i),
        isSame = false,
        "GetTupleElement on different index",
      ),
    )
  }

  def isTreeStructureSemanticallyEquivalent: ArraySeq[(BaseIR, BaseIR, Boolean, String)] =
    ArraySeq(
      (
        MakeArray(
          MakeArray(I32(0)),
          MakeArray(I32(0)),
        ),
        MakeArray(
          MakeArray(
            MakeArray(I32(0), I32(0))
          )
        ),
        false,
        "Tree structure contributes to semantics",
      )
    )

  def isValueIRSemanticallyEquivalent: IndexedSeq[(BaseIR, BaseIR, Boolean, String)] =
    IndexedSeq.concat(
      isTriviallySemanticallyEquivalent,
      isLetSemanticallyEquivalent,
      isBaseStructSemanticallyEquivalent,
      isTreeStructureSemanticallyEquivalent,
    )

  def isTableIRSemanticallyEquivalent: IndexedSeq[(BaseIR, BaseIR, Boolean, String)] = {
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

    IndexedSeq.concat(
      ArraySeq[(BaseIR, BaseIR, Boolean, String)](
        (tir, tir, true, "TableRead same table"),
        (tir, mkTableIR(ttype, "/another/fake/table"), false, "TableRead different table"),
        (
          TableKeyBy(tir, IndexedSeq("a")),
          TableKeyBy(tir, IndexedSeq("a")),
          true,
          "TableKeyBy same key",
        ),
        (
          TableKeyBy(tir, IndexedSeq("a")),
          TableKeyBy(tir, IndexedSeq("b")),
          false,
          "TableKeyBy different key",
        ),
      ),
      ArraySeq[String => TableReader](
        path =>
          new StringTableReader(
            StringTableReaderParameters(ArraySeq(path), None, false, false, false),
            new FakeFS().glob(path),
          ),
        path =>
          TableNativeZippedReader(
            path + ".left",
            path + ".right",
            None,
            mkFakeTableSpec(ttype),
            mkFakeTableSpec(ttypeb),
          ),
      )
        .map(mkTableRead _ compose _)
        .flatMap { reader =>
          ArraySeq[(BaseIR, BaseIR, Boolean, String)](
            (reader("/fake/table"), reader("/fake/table"), true, "read same table"),
            (
              reader("/fake/table"),
              reader("/another/fake/table"),
              false,
              "read different table",
            ),
          )
        },
      ArraySeq(
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
      ).map(wrap => (wrap(tir): BaseIR, wrap(tir): BaseIR, true, "")),
    )
  }

  def isBlockMatrixIRSemanticallyEquivalent: ArraySeq[(BaseIR, BaseIR, Boolean, String)] =
    ArraySeq[String => BlockMatrixReader](
      path => BlockMatrixBinaryReader(path, ArraySeq(1L, 1L), 1),
      path =>
        new BlockMatrixNativeReader(
          BlockMatrixNativeReaderParameters(path),
          BlockMatrixMetadata(1, 1, 1, None, IndexedSeq.empty),
        ),
    )
      .map(BlockMatrixRead compose _)
      .flatMap { reader =>
        Array[(BaseIR, BaseIR, Boolean, String)](
          (
            reader("/fake/block-matrix"),
            reader("/fake/block-matrix"),
            true,
            "Read same block matrix",
          ),
          (
            reader("/fake/block-matrix"),
            reader("/another/fake/block-matrix"),
            false,
            "Read different block matrix",
          ),
        )
      }

  def testSemanticEquivalence() =
    IndexedSeq.concat(
      isValueIRSemanticallyEquivalent,
      isTableIRSemanticallyEquivalent,
      isBlockMatrixIRSemanticallyEquivalent,
    )

  private[this] val NormalizeNames: (ExecuteContext, BaseIR) => BaseIR =
    ir.NormalizeNames(allowFreeVariables = true)

  @ParameterizedTest
  def testSemanticEquivalence(
    a: BaseIR,
    b: BaseIR,
    isEqual: Boolean,
    comment: String,
  )(implicit ctx: ExecuteContext
  ): Unit =
    ctx.local(fs = new FakeFS) { ctx =>
      val actual =
        SemanticHash(ctx, NormalizeNames(ctx, a)) == SemanticHash(ctx, NormalizeNames(ctx, b))
      assertEq(
        actual,
        isEqual,
        s"expected semhash($a) ${if (isEqual) "==" else "!="} semhash($b), $comment",
      )
    }

  @Test
  def testFileNotFoundExceptions(implicit ctx: ExecuteContext): Unit = {
    val fs =
      new FakeFS {
        override def eTag(url: FakeURL): Option[String] =
          throw new FileNotFoundException(url.path)
      }

    val ir = importMatrix("gs://fake-bucket/fake-matrix")

    ctx.local(fs = fs) { ctx =>
      assertEq(
        SemanticHash(ctx, ir),
        None,
        "SemHash should be resilient to FileNotFoundExceptions.",
      )
    }
  }

  class FakeFS extends is.hail.io.fs.FakeFS {
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
          override def matrix_type: MatrixType = ty
          override def references_rel_path: String = "references"
          override def globalsSpec: AbstractTableSpec = mkFakeTableSpec(ty.canonicalTableType)
          override def colsSpec: AbstractTableSpec = mkFakeTableSpec(ty.colsTableType)
          override def rowsSpec: AbstractTableSpec = mkFakeTableSpec(ty.rowsTableType)
          override def entriesSpec: AbstractTableSpec = mkFakeTableSpec(ty.entriesTableType)
          override def file_version: Int = 1
          override def hail_version: String = PrettyVersion
          override def components: Map[String, ComponentSpec] = Map.empty
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
