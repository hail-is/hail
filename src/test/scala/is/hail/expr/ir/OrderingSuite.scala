package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region, RegionValueBuilder}
import is.hail.check.{Gen, Prop}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import org.testng.annotations.Test
import is.hail.utils._
import org.apache.spark.sql.Row

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

      val ir = ApplySpecial("==", FastSeq(GetTupleElement(In(0, ttup), 0), GetTupleElement(In(1, ttup), 0)))
      val irfb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Boolean]

      Infer(ir)
      Emit(ir, irfb)

      val irf = irfb.result()()
      assert(irf(region, tup1, false, tup2, false) == (compare == 0))
      assert(irf(region, tup1, false, tup1, false))
      assert(irf(region, tup2, false, tup2, false))

      val ir2 = ApplySpecial("!=", FastSeq(GetTupleElement(In(0, TTuple(t)), 0), GetTupleElement(In(1, TTuple(t)), 0)))
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
}
