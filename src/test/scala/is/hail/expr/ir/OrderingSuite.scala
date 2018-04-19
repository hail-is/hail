package is.hail.expr.ir

import is.hail.annotations._
import is.hail.check.{Gen, Prop}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class OrderingSuite {

  @Test def testRandomAgainstUnsafe() {
    val compareGen = Type.genArb.flatMap(t => Gen.zip(Gen.const(t), t.genNonmissingValue, t.genNonmissingValue))
    val p = Prop.forAll(compareGen) { case (t, a1, a2) =>
      val region = Region()
      val rvb = new RegionValueBuilder(region)

      rvb.start(t)
      rvb.addAnnotation(t, a1)
      val v1 = rvb.end()

      rvb.start(t)
      rvb.addAnnotation(t, a2)
      val v2 = rvb.end()

      val compare = t.unsafeOrdering(true).compare(region, v1, region, v2)
      val fb = new EmitFunctionBuilder[AsmFunction3[Region, Long, Long, Int]](Array(GenericTypeInfo[Region](), GenericTypeInfo[Long](), GenericTypeInfo[Long]()), GenericTypeInfo[Int]())
      val stagedOrdering = new CodeOrdering(t, true)

      fb.emit(stagedOrdering.compare(fb.apply_method, fb.getArg[Region](1), fb.getArg[Long](2), fb.getArg[Region](1), fb.getArg[Long](3)))
      val f = fb.result()()
      f(region, v1, v2) == compare
    }
    p.check()
  }

  @Test def testEqualityInEmitFunction() {
    val compareGen = Type.genArb.flatMap(t => Gen.zip(Gen.const(t), t.genNonmissingValue, t.genNonmissingValue))
    val p = Prop.forAll(compareGen) { case (t, a1, a2) =>

      val ttup = TTuple(t)
      val region = Region()
      val rvb = new RegionValueBuilder(region)

      rvb.start(ttup)
      rvb.addAnnotation(ttup, Row(a1))
      val tup1 = rvb.end()

      rvb.start(ttup)
      rvb.addAnnotation(ttup, Row(a2))
      val tup2 = rvb.end()

      val v1 = ttup.loadField(region, tup1, 0)
      val v2 = ttup.loadField(region, tup2, 0)

      val compare = t.unsafeOrdering(true).compare(region, v1, region, v2)

      val ir = Apply("==", FastSeq(GetTupleElement(In(0, ttup), 0), GetTupleElement(In(1, ttup), 0)))
      val irfb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Boolean]

      Infer(ir)
      Emit(ir, irfb)

      val irf = irfb.result()()
      assert(irf(region, tup1, false, tup2, false) == (compare == 0))
      assert(irf(region, tup1, false, tup1, false))
      assert(irf(region, tup2, false, tup2, false))

      val ir2 = Apply("!=", FastSeq(GetTupleElement(In(0, TTuple(t)), 0), GetTupleElement(In(1, TTuple(t)), 0)))
      val irfb2 = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Boolean]

      Infer(ir2)
      Emit(ir2, irfb2)

      val irf2 = irfb2.result()()
      assert(irf2(region, tup1, false, tup2, false) == (compare != 0))
      assert(!irf2(region, tup1, false, tup1, false))
      assert(!irf2(region, tup2, false, tup2, false))
      true
    }
    p.check()
  }

  @Test def testSortOnRandomArray() {
    val compareGen = Type.genArb.flatMap(t => Gen.zip(Gen.const(t), TArray(t).genNonmissingValue))
    val p = Prop.forAll(compareGen) { case (t, a) =>
      val ir = ArraySort(GetTupleElement(In(0, TTuple(TArray(t))), 0))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]

      Infer(ir)
      Emit(ir, fb)

      val f = fb.result()()

      val region = Region()
      val rvb = new RegionValueBuilder(region)

      rvb.start(TTuple(TArray(t)))
      rvb.startTuple()
      rvb.addAnnotation(TArray(t), a)
      rvb.endTuple()
      val off = rvb.end()

      val res = f(region, off, false)
      val actual = SafeIndexedSeq(TArray(t), region, res)
      val expected = a.asInstanceOf[IndexedSeq[Any]].sorted(t.ordering.toOrdering)

      expected == actual
    }
    p.check()
  }

  @Test def testToSetOnRandomDuplicatedArray() {
    val compareGen = Type.genArb.flatMap(t => Gen.zip(Gen.const(t), TArray(t).genNonmissingValue))
    val p = Prop.forAll(compareGen) { case (t, a) =>
      val array = a.asInstanceOf[IndexedSeq[Any]] ++ a.asInstanceOf[IndexedSeq[Any]]
      val ir = Set(GetTupleElement(In(0, TTuple(TArray(t))), 0))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]

      Infer(ir)
      Emit(ir, fb)

      val f = fb.result()()

      val region = Region()
      val rvb = new RegionValueBuilder(region)

      rvb.start(TTuple(TArray(t)))
      rvb.startTuple()
      rvb.addAnnotation(TArray(t), array)
      rvb.endTuple()
      val off = rvb.end()

      val res = f(region, off, false)
      val actual = SafeIndexedSeq(TArray(t), region, res)
      val expected = a.asInstanceOf[IndexedSeq[Any]].sorted(t.ordering.toOrdering).distinct

      expected == actual
    }
    p.check()
  }


  @Test def testToDictOnRandomDuplicatedArray() {
    val compareGen = Gen.zip(Type.genArb, Type.genArb).flatMap {
      case (k, v) => Gen.zip(Gen.const(TTuple(k, v)), TArray(TTuple(k, v)).genNonmissingValue)
    }
    val p = Prop.forAll(compareGen) { case (telt, a) =>
      val array: IndexedSeq[Row] = a.asInstanceOf[IndexedSeq[Row]] ++ a.asInstanceOf[IndexedSeq[Row]]
      val ir = Dict(GetTupleElement(In(0, TTuple(TArray(telt))), 0))
      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]

      Infer(ir)
      Emit(ir, fb)

      val f = fb.result()()

      val region = Region()
      val rvb = new RegionValueBuilder(region)

      rvb.start(TTuple(TArray(telt)))
      rvb.startTuple()
      rvb.addAnnotation(TArray(telt), array)
      rvb.endTuple()
      val off = rvb.end()

      val res = f(region, off, false)
      val actual = SafeIndexedSeq(TArray(telt), region, res)
      val actualKeys = actual.filter(_ != null).map { case Row(k, _) => k }
      val expectedMap = array.filter(_ != null).map { case Row(k, v) => (k, v) }.toMap
      val expectedKeys = expectedMap.keys.toIndexedSeq.sorted(telt.types(0).ordering.toOrdering)

      expectedKeys == actualKeys
    }
    p.check()
  }
}
