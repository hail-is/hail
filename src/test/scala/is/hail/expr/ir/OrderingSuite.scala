package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValueBuilder, SafeIndexedSeq}
import is.hail.check.{Gen, Prop}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class OrderingSuite {

  def recursiveSize(t: Type): Int = {
    val inner = t match {
      case ti: TInterval => recursiveSize(ti.pointType)
      case tc: TContainer => recursiveSize(tc.elementType)
      case tbs: TBaseStruct =>
        tbs.types.map{ t => recursiveSize(t) }.sum
      case _ => 0
    }
    inner + 1
  }

  def getStagedOrderingFunction[T: TypeInfo](t: Type, comp: String): AsmFunction3[Region, Long, Long, T] = {
    val fb = EmitFunctionBuilder[Region, Long, Long, T]
    val stagedOrdering = t.codeOrdering(fb.apply_method)
    val cregion: Code[Region] = fb.getArg[Region](1)
    val cv1 = coerce[stagedOrdering.T](cregion.getIRIntermediate(t)(fb.getArg[Long](2)))
    val cv2 = coerce[stagedOrdering.T](cregion.getIRIntermediate(t)(fb.getArg[Long](3)))
    comp match {
      case "compare" => fb.emit(stagedOrdering.compare(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "equiv" => fb.emit(stagedOrdering.equiv(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "lt" => fb.emit(stagedOrdering.lt(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "lteq" => fb.emit(stagedOrdering.lteq(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "gt" => fb.emit(stagedOrdering.gt(cregion, (const(false), cv1), cregion, (const(false), cv2)))
      case "gteq" => fb.emit(stagedOrdering.gteq(cregion, (const(false), cv1), cregion, (const(false), cv2)))
    }

    fb.result()()
  }

  @Test def testRandomOpsAgainstExtended() {
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

      val compare = java.lang.Integer.signum(t.ordering.compare(a1, a2))
      val fcompare = getStagedOrderingFunction[Int](t, "compare")
      val result = java.lang.Integer.signum(fcompare(region, v1, v2))

      assert(result == compare, s"compare expected: $compare vs $result")


      val equiv = t.ordering.equiv(a1, a2)
      val fequiv = getStagedOrderingFunction[Boolean](t, "equiv")

      assert(fequiv(region, v1, v2) == equiv, s"equiv expected: $equiv")

      val lt = t.ordering.lt(a1, a2)
      val flt = getStagedOrderingFunction[Boolean](t, "lt")

      assert(flt(region, v1, v2) == lt, s"lt expected: $lt")

      val lteq = t.ordering.lteq(a1, a2)
      val flteq = getStagedOrderingFunction[Boolean](t, "lteq")

      assert(flteq(region, v1, v2) == lteq, s"lteq expected: $lteq")

      val gt = t.ordering.gt(a1, a2)
      val fgt = getStagedOrderingFunction[Boolean](t, "gt")

      assert(fgt(region, v1, v2) == gt, s"gt expected: $gt")

      val gteq = t.ordering.gteq(a1, a2)
      val fgteq = getStagedOrderingFunction[Boolean](t, "gteq")

      assert(fgteq(region, v1, v2) == gteq, s"gteq expected: $gteq")

      true
    }
    p.check()
  }
}
