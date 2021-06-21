package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.TestUtils.assertEvalsTo
import is.hail.types.virtual.{TArray, TBoolean, TFloat32, TFloat64, TInt32, TTuple}
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class FoldConstantsSuite extends HailSuite {
  @Test def testRandomBlocksFolding() {
    val x = ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64)
    assert(FoldConstants(ctx, x) == x)
  }

  @Test def testErrorCatching() {
    val ir = invoke("toInt32", TInt32, Str(""))
    assert(FoldConstants(ctx, ir) == ir)
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
    val i4 = I32(4)
    val i8 = I32(8)
    val ref = "x"
    val vRef = "y"
    val range = StreamRange(0, 10, 1)
    val refIR = Ref(ref, TInt32)
    val vRefIR = Ref(vRef, TInt32)
    val streamIRA = ToStream(Literal(TArray(TInt32), FastIndexedSeq(12, 2, 4, refIR)))
    val streamIRB = ToStream(Literal(TArray(TInt32), FastIndexedSeq(1, vRefIR, 2, 4)))
    val behavior = ArrayZipBehavior.TakeMinLength
    val toZip = IndexedSeq(streamIRA, streamIRB)
    val zipNames = IndexedSeq("A", "B")
    val oppIR = ApplyBinaryPrimOp(Add(), i4, i8)
    val refOppIR = ApplyBinaryPrimOp(Add(), refIR, i8)

    val MakeStreamIR = MakeStream.unify(FastIndexedSeq(12, 2, 4, 8, 6).zipWithIndex.map
                       { case (n, idx) => MakeStruct(FastIndexedSeq("lk" -> (if (n == null)
                         NA(TInt32) else I32(n)), "l" -> I32(idx)))})

    val streamFoldBodyIR = ApplyBinaryPrimOp(Add(), vRefIR, refOppIR)
    val letIR = Let(ref, i4, refOppIR)
    val streamMapIR = StreamMap(range, ref, refOppIR)
    val streamFoldIR = StreamFold(range, Ref("z", TInt32), ref, vRef, streamFoldBodyIR)
    val streamFoldLetIR = Let("z", i4, streamFoldIR)
    val letZipIR = Let(ref, i8, Let(vRef, i4, StreamZip(toZip, zipNames, True(), behavior)))
    val streamFold2IR = Let(ref, i8, Let(vRef, i4, StreamFold2(streamIRA,
                        FastIndexedSeq((ref, I32(0)), (vRef, I32(8))),
                        "val", FastIndexedSeq(Ref("val", TInt32) + Ref(ref, TInt32),
                        Coalesce(FastSeq(Ref(vRef, TInt32), Ref("val", TInt32)))),
                        MakeStruct(FastSeq((ref, Ref(ref, TInt32)), (vRef, Ref(vRef, TInt32)))))))


    val toArrayStreamFilterIR = ToArray(
      StreamFilter(
        StreamMap(
          range,
          "x", ApplyBinaryPrimOp(Add(), Ref("x", TInt32), I32(3))),
        "element",
        ApplyComparisonOp(LT(TInt32, TInt32), Ref("element", TInt32), ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64))))

    val makeTupleSeededIR = MakeTuple.ordered(Seq(ToArray(StreamRange(I32(3), I32(8), I32(1))), ApplySeeded("rand_norm", Seq(F64(0d),
                             F64(0d)), 0L, TFloat64), Let("y", I32(4), Let ("x", ApplyBinaryPrimOp(Add(), I32(3), Ref("y", TInt32)),
                               MakeTuple.ordered(Seq(ApplyBinaryPrimOp(Add(), Ref("x", TInt32), Ref("y", TInt32))))))))

    val randLetIR = Let("y", ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64),
                    Let("x", ApplyBinaryPrimOp(Add(), I32(1), I32(2)) ,
                      ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), Ref("x", TInt32), 1),
                        ApplyBinaryPrimOp(Multiply(),Ref("x", TInt32), Ref("y", TInt32)))))

    val absIR = Apply("abs", Seq(), Seq(I32(-5)), TInt32)

    val errorIR = Let("x", ToArray(StreamRange(0, 10, 1)),
                      If(ApplyComparisonOp(LT(TInt32, TInt32), ArrayLen(Ref("x", TArray(TInt32))), I32(1)),
                      ApplyBinaryPrimOp(Add(), I32(3), ArrayRef(Ref("x", TArray(TInt32)), I32(30))),
                      ApplyBinaryPrimOp(Add(), I32(2), ArrayRef(Ref("x", TArray(TInt32)), I32(30)))))





    (FoldConstants(ctx, toArrayStreamFilterIR))
  //  println(FoldConstants(ctx, makeTupleSeededIR))
    //  println(FoldConstants(ctx, randLetIR))

//    ))

//    val oppTest = FoldConstants.findConstantSubTrees(oppIR)
//    assert(oppTest.contains(oppIR))
//
//    val letTest = FoldConstants.findConstantSubTrees(letIR)
//    assert(letTest.contains(letIR))
//
//    val streamMapTest = FoldConstants.findConstantSubTrees(streamMapIR)
//    assert(streamMapTest.contains(streamMapIR))
//
//    val streamFoldTest = FoldConstants.findConstantSubTrees(streamFoldLetIR)
//    assert(streamFoldTest.contains(streamFoldLetIR))
//
//    val zipTest = FoldConstants.findConstantSubTrees(letZipIR)
//    assert(zipTest.contains(letZipIR))
//
//    val streamFold2Test = FoldConstants.findConstantSubTrees(streamFold2IR)
//    assert(streamFold2Test.contains(streamFold2IR))
    //print(CompileAndEvaluate(ctx, errorIR))
    //println(FoldConstants(ctx, boolIR))
  }
}
