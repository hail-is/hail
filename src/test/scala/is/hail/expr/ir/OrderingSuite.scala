package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region, RegionValueBuilder}
import is.hail.check.{Gen, Prop}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils.Interval
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
}
