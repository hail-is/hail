package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.TestUtils.assertEvalsTo
import is.hail.types.virtual.{TArray, TBoolean, TFloat32, TFloat64, TInt32, TStream, TStruct, TTuple, Type}
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

import scala.collection.mutable.ArrayBuffer

class FoldConstantsSuite extends HailSuite {
  @Test def testRandomBlocksFolding() {
    val x = ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64)
    assert(FoldConstants(ctx, x) == x)
  }

  @Test def testErrorCatching() {
    val ir = invoke("toInt32", TInt32, Str(""))
    val errorCompiled = FoldConstants(ctx, ir)
    errorCompiled match {
      case Die(Str(str), typ, id: Int) =>
        assert(typ == TInt32)
        assert(id == -1)
        println(str)
        assert(str.contains("Could not parse '' as Int32"))
      case _ => fail()
    }
  }

  @DataProvider(name = "aggNodes")
  def aggNodes(): Array[Array[Any]] = {
    Array[IR](
      AggLet("x", I32(1), I32(1), false),
      AggLet("x", I32(1), I32(1), true),
      ApplyAggOp(Sum())(I64(1)),
      ApplyScanOp(Sum())(I64(1))
      ).map(x => Array[Any](x))
  }

  @Test def testAggNodesConstruction(): Unit = aggNodes()

  @Test(dataProvider = "aggNodes") def testAggNodesDoNotFold(node: IR): Unit = {
    assert(FoldConstants(ctx, node) == node)
  }

  @Test def testFindConstantSubtrees(): Unit = {
    val streamIRA = ToStream(Literal(TArray(TInt32), FastIndexedSeq(12, 2, 4, 5)))
    val streamIRB = ToStream(Literal(TArray(TInt32), FastIndexedSeq(1, Ref("x", TInt32), 2, Ref("y", TInt32))))
    val toZip = IndexedSeq(streamIRA, streamIRB)
    val zipNames = IndexedSeq("A", "B")

    val makeStreamIR = MakeStream.unify(FastIndexedSeq(12, 2, 4, 8, 6).zipWithIndex.map
                       { case (n, idx) => MakeStruct(FastIndexedSeq("lk" -> (if (n == null)
                         NA(TInt32) else I32(n)), "l" -> I32(idx)))})
    val makeStreamIRConst = MakeStream(ArrayBuffer(Literal(TStruct(("lk", TInt32),("l", TInt32)), Row(12,0)),
                                                  Literal(TStruct(("lk", TInt32),("l", TInt32)), Row(2,1)),
                                                  Literal(TStruct(("lk", TInt32),("l", TInt32)), Row(4,2)),
                                                  Literal(TStruct(("lk", TInt32),("l", TInt32)), Row(8,3)),
                                                  Literal(TStruct(("lk", TInt32),("l", TInt32)), Row(6,4))),
                                                  TStream(TStruct(("lk", TInt32),("l", TInt32))))
    val toArrayStreamFilterIR = ToArray(
                                  StreamFilter(StreamMap( streamIRA, "x",
                                    ApplyBinaryPrimOp(Add(), Ref("x", TInt32), I32(3))), "element",
                                      ApplyComparisonOp(LT(TInt32, TInt32),
                                        Ref("element", TInt32),
                                        ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64))))
    val makeTupleSeededIR = MakeTuple.ordered(Seq(ToArray
                                                (StreamRange(I32(3), I32(8), I32(1))),
                                                ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64),
                                                Let("y", I32(4),
                                                  Let ("x", ApplyBinaryPrimOp(Add(), I32(3), Ref("y", TInt32)),
                                                       MakeTuple.ordered(
                                                         Seq(ApplyBinaryPrimOp(Add(), Ref("x", TInt32),
                                                                              Ref("y", TInt32))))))))
    val makeTupleSeededIRConst = MakeTuple.ordered(
                                  ArrayBuffer(Literal(TArray(TInt32), Seq(3,4,5,6,7)),
                                              ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64),
                                              Literal(TTuple(TInt32), Row(11))))
    val randLetIR = Let("y",
                        ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64),
                        Let("x", ApplyBinaryPrimOp(Add(), I32(1), I32(2)) ,
                            ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), Ref("x", TInt32), 1),
                                              ApplyBinaryPrimOp(Multiply(),Ref("x", TInt32), Ref("y", TInt32)))))
    val randLetIRConst = Let("y",
                             ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64),
                             Let("x", I32(3),
                               ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), Ref("x", TInt32), 1),
                                 ApplyBinaryPrimOp(Multiply(),Ref("x", TInt32), Ref("y", TInt32)))))
    val errorIR =
                      If(
                        ApplyComparisonOp(LT(TInt32, TInt32),
                                          ArrayLen(Literal(TArray(TInt32), FastIndexedSeq(0, 1, 2))),
                                          I32(1)),
                        ApplyBinaryPrimOp(Add(), F64(3d),
                                          ArrayRef(Literal(TArray(TFloat64), FastIndexedSeq(0d, 1d, 2d)), I32(-1))),
                        ApplyBinaryPrimOp(Add(),
                                          ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64), F64(22d)))


    assert(FoldConstants(ctx, makeStreamIR) == makeStreamIRConst)
    assert(FoldConstants(ctx, toArrayStreamFilterIR) == toArrayStreamFilterIR)
    assert(FoldConstants(ctx, makeTupleSeededIR) == makeTupleSeededIRConst)
    assert(FoldConstants(ctx, randLetIR) == randLetIRConst)

    val errorCompiled = FoldConstants(ctx, errorIR)
    errorCompiled match {
      case If(cond, Die(Str(str), typ, id: Int), _) =>
        assert(cond == False())
        assert(typ == TFloat64)
        assert(id == -1)
        assert(str.contains("array index out of bounds"))
      case _ => fail()
    }
  }
}
