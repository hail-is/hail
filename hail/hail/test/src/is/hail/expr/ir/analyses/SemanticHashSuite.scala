package is.hail.expr.ir.analyses

import is.hail.{HAIL_PRETTY_VERSION, HailSuite}
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.io.fs.{FS, FakeFS, FakeURL, FileListEntry}
import is.hail.linalg.BlockMatrixMetadata
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.virtual._
import is.hail.utils.FastSeq

import java.io.FileNotFoundException
import java.lang

import org.json4s.JValue
import org.scalatest
import org.testng.annotations.{DataProvider, Test}

class SemanticHashSuite extends HailSuite {

  def isTriviallySemanticallyEquivalent: Array[Array[Any]] =
    Array(
      Array(True(), True(), true, "Refl"),
      Array(False(), False(), true, "Refl"),
      Array(True(), False(), false, "Refl"),
      Array(I32(0), I32(0), true, "Refl"),
      Array(I32(0), I32(1), false, "Refl"),
      Array(I64(0), I64(0), true, "Refl"),
      Array(I64(0), I64(1), false, "Refl"),
      Array(F32(0), F32(0), true, "Refl"),
      Array(F32(0), F32(1), false, "Refl"),
      Array(Void(), Void(), true, "Refl"),
      Array(Str("a"), Str("a"), true, "Refl"),
      Array(Str("a"), Str("b"), false, "Refl"),
      Array(NA(TInt32), NA(TInt32), true, "Refl"),
      Array(NA(TInt32), NA(TFloat64), false, "Refl"),
    )

  def mkRelationalLet(bindings: IndexedSeq[(Name, IR)], body: IR): IR =
    bindings.foldRight(body) { case ((name, value), body) =>
      RelationalLet(name, value, body)
    }

  def isLetSemanticallyEquivalent: Array[Array[Any]] = {
    val x = freshName()
    val y = freshName()
    Array((Let(_, _), Ref), (mkRelationalLet _, RelationalRef)).flatMap { case (let, ref) =>
      Array(
        Array(
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(y -> I32(0)), ref(y, TInt32)),
          true,
          "names used in let-bindings do not change semantics",
        ),
        Array(
          let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
          let(FastSeq(y -> I32(0), x -> I32(0)), ref(y, TInt32)),
          true,
          "names of let-bindings do not change semantics",
        ),
        Array(
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(x -> I64(0)), ref(x, TInt64)),
          false,
          "different IRs",
        ),
        Array(
          let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
          let(FastSeq(y -> I32(0), x -> I32(0)), ref(x, TInt32)),
          false,
          "Different binding being referenced",
        ),
        /* `SemanticHash` does not perform or recognise opportunities for simplification.
         * The following examples demonstrate some of its limitations as a consequence. */
        Array(
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(x -> let(FastSeq(freshName() -> I32(0)), I32(0))), ref(x, TInt32)),
          false,
          "SemanticHash does not simplify",
        ),
        Array(
          let(FastSeq(x -> I32(0)), ref(x, TInt32)),
          let(FastSeq(x -> I32(0), y -> I32(0)), ref(x, TInt32)),
          false,
          "SemanticHash does not simplify",
        ),
      )
    }
  }

  def isBaseStructSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      Array(
        Array(
          MakeStruct(Array.empty[(String, IR)]),
          MakeStruct(Array.empty[(String, IR)]),
          true,
          "empty structs",
        ),
        Array(
          MakeStruct(Array(genUID() -> I32(0))),
          MakeStruct(Array(genUID() -> I32(0))),
          true,
          "field names do not affect MakeStruct semantics",
        ),
        Array(
          MakeTuple(Array.empty[(Int, IR)]),
          MakeTuple(Array.empty[(Int, IR)]),
          true,
          "empty tuples",
        ),
        Array(
          MakeTuple(Array(0 -> I32(0))),
          MakeTuple(Array(0 -> I32(0))),
          true,
          "identical tuples",
        ),
        Array(
          MakeTuple(Array(0 -> I32(0))),
          MakeTuple(Array(1 -> I32(0))),
          false,
          "tuple indices affect MakeTuple semantics",
        ),
      ), {

        def f(mkType: Int => Type, get: (IR, Int) => IR, isSame: Boolean, reason: String) =
          Array.tabulate(2)(idx => bindIR(NA(mkType(idx)))(get(_, idx))) ++ Array(isSame, reason)

        Array(
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
            mkType = i => TTuple(Array(TupleField(i, TInt32))),
            get = (ir, i) => GetTupleElement(ir, i),
            isSame = false,
            "GetTupleElement on different index",
          ),
        )
      },
    )

  def isTreeStructureSemanticallyEquivalent: Array[Array[Any]] =
    Array(
      Array(
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

  def isValueIRSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      isTriviallySemanticallyEquivalent,
      isLetSemanticallyEquivalent,
      isBaseStructSemanticallyEquivalent,
      isTreeStructureSemanticallyEquivalent,
    )

  def isTableIRSemanticallyEquivalent: Array[Array[Any]] = {
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

    Array.concat(
      Array(
        Array(tir, tir, true, "TableRead same table"),
        Array(tir, mkTableIR(ttype, "/another/fake/table"), false, "TableRead different table"),
        Array(
          TableKeyBy(tir, IndexedSeq("a")),
          TableKeyBy(tir, IndexedSeq("a")),
          true,
          "TableKeyBy same key",
        ),
        Array(
          TableKeyBy(tir, IndexedSeq("a")),
          TableKeyBy(tir, IndexedSeq("b")),
          false,
          "TableKeyBy different key",
        ),
      ),
      Array[String => TableReader](
        path =>
          new StringTableReader(
            StringTableReaderParameters(Array(path), None, false, false, false),
            fakeFs.glob(path),
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
          Array(
            Array(reader("/fake/table"), reader("/fake/table"), true, "read same table"),
            Array(
              reader("/fake/table"),
              reader("/another/fake/table"),
              false,
              "read different table",
            ),
          )
        },
      Array(
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
      ).map(wrap => Array(wrap(tir), wrap(tir), true, "")),
    )
  }

  def isBlockMatrixIRSemanticallyEquivalent: Array[Array[Any]] =
    Array[String => BlockMatrixReader](
      path => BlockMatrixBinaryReader(path, Array(1L, 1L), 1),
      path =>
        new BlockMatrixNativeReader(
          BlockMatrixNativeReaderParameters(path),
          BlockMatrixMetadata(1, 1, 1, None, IndexedSeq.empty),
        ),
    )
      .map(BlockMatrixRead compose _)
      .flatMap { reader =>
        Array(
          Array(
            reader("/fake/block-matrix"),
            reader("/fake/block-matrix"),
            true,
            "Read same block matrix",
          ),
          Array(
            reader("/fake/block-matrix"),
            reader("/another/fake/block-matrix"),
            false,
            "Read different block matrix",
          ),
        )
      }

  @DataProvider(name = "isBaseIRSemanticallyEquivalent")
  def isBaseIRSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      isValueIRSemanticallyEquivalent,
      isTableIRSemanticallyEquivalent,
      isBlockMatrixIRSemanticallyEquivalent,
    )

  @Test(dataProvider = "isBaseIRSemanticallyEquivalent")
  def testSemanticEquivalence(a: BaseIR, b: BaseIR, isEqual: Boolean, comment: String)
    : scalatest.Assertion =
    ctx.local(fs = fakeFs) { ctx =>
      assertResult(
        isEqual,
        s"expected semhash($a) ${if (isEqual) "==" else "!="} semhash($b), $comment",
      )(
        SemanticHash(ctx, a) == SemanticHash(ctx, b)
      )
    }

  @Test
  def testFileNotFoundExceptions(): scalatest.Assertion = {
    val fs =
      new FakeFS {
        override def eTag(url: FakeURL): Option[String] =
          throw new FileNotFoundException(url.getPath)
      }

    val ir = importMatrix("gs://fake-bucket/fake-matrix")

    ctx.local(fs = fs) { ctx =>
      assertResult(None, "SemHash should be resilient to FileNotFoundExceptions.")(
        SemanticHash(ctx, ir)
      )
    }
  }

  private[this] val fakeFs: FS =
    new FakeFS {
      override def eTag(url: FakeURL): Option[String] =
        Some(url.getPath)

      override def glob(url: FakeURL): Array[FileListEntry] =
        Array(new FileListEntry {
          override def getPath: String = url.getPath
          override def getActualUrl: String = url.getPath
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
            HAIL_PRETTY_VERSION
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
        Map("partition_counts" -> PartitionCountsComponentSpec(Array(1L)))

      override def toJValue: JValue = ???
    }
}
