package is.hail.expr.ir.analyses

import is.hail.expr.ir._
import is.hail.expr.ir.analyses.SemanticHash.Hash
import is.hail.io.FakeFS
import is.hail.io.fs.FS
import is.hail.types.TableType
import is.hail.types.virtual._
import org.scalatest.Assertions.assertResult
import org.testng.annotations.{DataProvider, Test}

import scala.util.Random

class SemanticHashSuite {

  def isTableIRSemanticallyEquivalent: Array[Array[Any]] = {
    def mkTableIR(ttype: TableType, path: String): TableIR =
      TableRead(ttype, dropRows = false, new FakeTableReader() {
        override def pathsUsed: Seq[String] = IndexedSeq(path)

        override def fullType: TableType = ttype
      })

    val ttype = TableType(TStruct("a" -> TInt32, "b" -> TStruct()), IndexedSeq(), TStruct())
    val tir = mkTableIR(ttype, "gs://fake-bucket/fake-table")

    Array.concat(
      Array(
        Array(TableKeyBy(tir, IndexedSeq("a")), TableKeyBy(tir, IndexedSeq("a")), true),
        Array(TableKeyBy(tir, IndexedSeq("a")), TableKeyBy(tir, IndexedSeq("b")), false)
      ),
      Array(
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
    )
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

  def isMakeBaseStructSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      Array(
        Array(MakeStruct(Array.empty[(String, IR)]), MakeStruct(Array.empty[(String, IR)]), true),
        Array(MakeStruct(Array(genUID() -> Void())), MakeStruct(Array(genUID() -> Void())), true),
        Array(MakeTuple(Array.empty[(Int, IR)]), MakeTuple(Array.empty[(Int, IR)]), true),
        Array(MakeTuple(Array(0 -> Void())), MakeTuple(Array(0 -> Void())), true),
        Array(MakeTuple(Array(Random.nextInt -> Void())), MakeTuple(Array(Random.nextInt -> Void())), false)
      ), {
        def f(mkType: Int => Type, get: (IR, Int) => IR, isSame: Boolean) =
          Array.tabulate(2) { idx => bindIR(NA(mkType(idx)))(get(_, idx)) } ++ Array(isSame)

        Array(
          f(mkType = i => TStruct(i.toString -> TVoid), get = (ir, i) => GetField(ir, i.toString), isSame = true),
          f(mkType = _ => TTuple(TVoid), get = (ir, _) => GetTupleElement(ir, 0), isSame = true),
          f(mkType = i => TTuple(Array(TupleField(i, TVoid))), get = (ir, i) => GetTupleElement(ir, i), isSame = false)
        )
      }
    )

  @DataProvider(name = "isBaseIRSemanticallyEquivalent")
  def isBaseIRSemanticallyEquivalent: Array[Array[Any]] =
    Array.concat(
      isTableIRSemanticallyEquivalent,
      isLetSemanticallyEquivalent,
      isMakeBaseStructSemanticallyEquivalent
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
      override def fileChecksum(filename: String): Array[Byte] =
        filename.getBytes
    }
}
