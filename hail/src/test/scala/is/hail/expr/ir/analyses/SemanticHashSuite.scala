package is.hail.expr.ir.analyses

import is.hail.expr.ir._
import is.hail.expr.ir.analyses.SemanticHash.Hash
import is.hail.io.FakeFS
import is.hail.io.fs.{FS, FileStatus}
import is.hail.types.TableType
import is.hail.types.virtual.{TInt32, TStruct}
import org.scalatest.Assertions.assertResult
import org.testng.annotations.{DataProvider, Test}

import java.lang

class SemanticHashSuite {

  def isTableIRSemanticallyEquivalent: Array[Array[Any]] = {
    def mkTableIR(ttype: TableType, path: String): TableIR =
      TableRead(ttype, dropRows = false, new FakeTableReader() {
        override def pathsUsed: Seq[String] = IndexedSeq(path)

        override def fullType: TableType = ttype
      })

    val ttype = TableType(TStruct("a" -> TInt32, "b" -> TStruct()), IndexedSeq(), TStruct())
    val tir = mkTableIR(ttype, "gs://fake-bucket/fake-table")

    Array(
      Array(TableKeyBy(tir, IndexedSeq("a")), TableKeyBy(tir, IndexedSeq("a")), true),
      Array(TableKeyBy(tir, IndexedSeq("a")), TableKeyBy(tir, IndexedSeq("b")), false)
    ) ++ Array(
      TableGetGlobals,
      TableCollect,
      TableAggregate(_, Void()),
      TableCount,
      TableMapRows(_, MakeStruct.empty),
      TableMapGlobals(_, MakeStruct.empty),
      TableFilter(_, Void()),
      TableDistinct
    ).flatMap { wrap =>
      Array(
        Array(wrap(tir), wrap(tir), true),
        Array(wrap(tir), wrap(mkTableIR(ttype, "/fake/table")), false)
      )
    }
  }

  def isLetSemanticallyEquivalent: Array[Array[Any]] = {
    val input = Void()
    Array(Let, RelationalLet).flatMap { let =>
      Array(
        // let-bound names don't change the semantics
        Array(let("A", input, Ref("A", input.typ)), let("B", input, Ref("B", input.typ)), true),

        // if some other operation
        Array(let("A", input, let("B", Void(), Ref("A", input.typ))), let("B", input, Ref("B", input.typ)), false),

        // For simplicity, we assume copy propagation has occurred by this point.
        Array(
          let("A", input, let("B", Ref("A", input.typ), Ref("B", input.typ))),
          let("A", input, Ref("A", input.typ)),
          false
        )
      )
    }
  }

  @DataProvider(name = "isBaseIRSemanticallyEquivalent")
  def isBaseIRSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      isTableIRSemanticallyEquivalent,
      isLetSemanticallyEquivalent
    )


  @Test(dataProvider = "isBaseIRSemanticallyEquivalent")
  def testSemanticEquivalence(a: BaseIR, b: BaseIR, isEqual: Boolean): Unit =
    assertResult(isEqual, s"expected semhash($a) ${if (isEqual) "==" else "!="} semhash($b)")(
      semhash(a) == semhash(b)
    )


  val semhash: BaseIR => Hash.Type =
    ir => SemanticHash(fakeFs)(ir)._1


  val fakeFs: FS =
    new FakeFS {
      override def fileStatus(filename: String): FileStatus =
        new FileStatus {
          override def getPath: String = filename
          override def getModificationTime: lang.Long = filename.length.toLong
          override def getLen: Long = filename.length.toLong
          override def isDirectory: Boolean = false
          override def isSymlink: Boolean = false
          override def isFile: Boolean = true
          override def getOwner: String = "me@fun.is"
        }
    }
}
