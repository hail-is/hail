package is.hail.expr.ir.analyses

import is.hail.expr.ir._
import is.hail.io.FakeFS
import is.hail.io.fs.{FS, FileStatus, HadoopFS, LocalFSURL}
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.TableType
import is.hail.types.virtual._
import org.json4s.JValue
import org.scalatest.Assertions.assertResult
import org.testng.annotations.{DataProvider, Test}

import java.lang
import scala.util.Random

class SemanticHashSuite {

  def isTriviallySemanticallyEquivalent: Array[Array[Any]] =
    Array(
      Array(    True(),       True(),  true, "Refl"),
      Array(   False(),      False(),  true, "Refl"),
      Array(    True(),      False(), false, "Refl"),
      Array(    I32(0),       I32(0),  true, "Refl"),
      Array(    I32(0),       I32(1), false, "Refl"),
      Array(    I64(0),       I64(0),  true, "Refl"),
      Array(    I64(0),       I64(1), false, "Refl"),
      Array(    F32(0),       F32(0),  true, "Refl"),
      Array(    F32(0),       F32(1), false, "Refl"),
      Array(    Void(),       Void(),  true, "Refl"),
      Array(  Str("a"),     Str("a"),  true, "Refl"),
      Array(  Str("a"),     Str("b"), false, "Refl"),
      Array(NA(TInt32),   NA(TInt32),  true, "Refl"),
      Array(NA(TInt32), NA(TFloat64), false, "Refl")
    )

  def isLetSemanticallyEquivalent: Array[Array[Any]] =
    Array((Let, Ref), (RelationalLet, RelationalRef)).flatMap { case (let, ref) =>
      Array(
        Array(
          let("A", Void(), ref("A", TVoid)),
          let("B", Void(), ref("B", TVoid)),
          true,
          "names used in let-bindings do not change semantics"
        ),
        Array(
          let("A", I32(0), ref("A", TInt32)),
          let("B", Void(), ref("B", TVoid)),
          false,
          "different IRs"
        ),
        /* `SemanticHash` does not perform or recognise opportunities for simplification.
         * The following examples demonstrate some of its limitations as a consequence.
         */
        Array(
          let("A", Void(), ref("A", TVoid)),
          let("A", let(genUID(), I32(0), Void()), ref("A", TVoid)),
          false,
          "SemanticHash does not simplify"
        ),
        Array(
          let("A", Void(), ref("A", TVoid)),
          let("A", Void(), let("B", I32(0), ref("A", TVoid))),
          false,
          "SemanticHash does not simplify"
        )
      )
    }

  def isBaseStructSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      Array(
        Array(
          MakeStruct(Array.empty[(String, IR)]),
          MakeStruct(Array.empty[(String, IR)]),
          true,
          "empty structs"
        ),
        Array(
          MakeStruct(Array(genUID() -> Void())),
          MakeStruct(Array(genUID() -> Void())),
          true,
          "field names do not affect MakeStruct semantics"
        ),
        Array(
          MakeTuple(Array.empty[(Int, IR)]),
          MakeTuple(Array.empty[(Int, IR)]),
          true,
          "empty tuples"
        ),
        Array(
          MakeTuple(Array(0 -> Void())),
          MakeTuple(Array(0 -> Void())),
          true,
          "identical tuples"
        ),
        Array(
          MakeTuple(Array(0 -> Void())),
          MakeTuple(Array(1 -> Void())),
          false,
          "tuple indices affect MakeTuple semantics"
        )
      ), {

        def f(mkType: Int => Type, get: (IR, Int) => IR, isSame: Boolean, reason: String) =
          Array.tabulate(2) { idx => bindIR(NA(mkType(idx)))(get(_, idx)) } ++ Array(isSame, reason)

        Array(
          f(
            mkType = i => TStruct(i.toString -> TVoid),
            get = (ir, i) => GetField(ir, i.toString),
            isSame = true,
            "field names do not affect GetField semantics"
          ),
          f(
            mkType = _ => TTuple(TVoid),
            get = (ir, _) => GetTupleElement(ir, 0),
            isSame = true,
            "GetTupleElement of same index"
          ),
          f(
            mkType = i => TTuple(Array(TupleField(i, TVoid))),
            get = (ir, i) => GetTupleElement(ir, i),
            isSame = false,
            "GetTupleElement on different index"
          )
        )
      }
    )

  def isValueIRSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      isTriviallySemanticallyEquivalent,
      isLetSemanticallyEquivalent,
      isBaseStructSemanticallyEquivalent
    )

  def isTableIRSemanticallyEquivalent: Array[Array[Any]] = {
    def mkTableIR(ttype: TableType, path: String): TableIR =
      TableRead(ttype, dropRows = false, new TableNativeReader(
        TableNativeReaderParameters(path, None),
        new AbstractTableSpec {
          override def references_rel_path: String = ???
          override def table_type: TableType = ttype
          override def rowsSpec: AbstractRVDSpec = ???
          override def globalsSpec: AbstractRVDSpec = ???
          override def file_version: Int = ???
          override def hail_version: String = ???
          override def components: Map[String, ComponentSpec] = ???
          override def toJValue: JValue = ???
        }
      ))

    val ttype = TableType(TStruct("a" -> TInt32, "b" -> TStruct()), IndexedSeq(), TStruct())
    val tir = mkTableIR(ttype, "/fake/table")

    Array.concat(
      Array(
        Array(tir, tir, true, "TableRead same table"),
        Array(tir, mkTableIR(ttype, "/another/fake/table"), false, "TableRead different table"),
        Array(TableKeyBy(tir, IndexedSeq("a")), TableKeyBy(tir, IndexedSeq("a")), true, "TableKeyBy same key"),
        Array(TableKeyBy(tir, IndexedSeq("a")), TableKeyBy(tir, IndexedSeq("b")), false, "TableKeyBy different key")
      ),
      Array(
        TableGetGlobals,
        TableAggregate(_, Void()),
        TableCollect,
        TableCount,
        TableDistinct,
        TableFilter(_, Void()),
        TableMapGlobals(_, MakeStruct(IndexedSeq.empty)),
        TableMapRows(_, MakeStruct(IndexedSeq.empty)),
        TableRename(_, Map.empty, Map.empty),
      ).map { wrap =>
        Array(wrap(tir), wrap(tir), true, "")
      }
    )
  }

  def isMatrixIRSemanticallyEquivalent: Array[Array[Any]] =
    Array.empty

  @DataProvider(name = "isBaseIRSemanticallyEquivalent")
  def isBaseIRSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      isValueIRSemanticallyEquivalent,
      isTableIRSemanticallyEquivalent,
      isMatrixIRSemanticallyEquivalent
    )

  @Test(dataProvider = "isBaseIRSemanticallyEquivalent")
  def testSemanticEquivalence(a: BaseIR, b: BaseIR, isEqual: Boolean, comment: String): Unit =
    assertResult(isEqual, s"expected semhash($a) ${if (isEqual) "==" else "!="} semhash($b), $comment")(
      semhash(a) == semhash(b)
    )


  val semhash: BaseIR => Option[SemanticHash.Type] =
    ir => SemanticHash(fakeFs)(ir)


  val fakeFs: FS =
    new FakeFS {
      override def fileChecksum(filename: String): Array[Byte] =
        filename.getBytes

      override def glob(filename: String): Array[FileStatus] =
        Array(new FileStatus {
          override def getPath: String = filename
          override def getModificationTime: lang.Long = ???
          override def getLen: Long = ???
          override def isDirectory: Boolean = ???
          override def isSymlink: Boolean = ???
          override def isFile: Boolean = true
          override def getOwner: String = ???
        })
    }
}