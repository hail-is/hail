package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, RegionValue, SafeRow, StagedRegionValueBuilder}
import is.hail.asm4s.Code
import is.hail.expr.ir.EmitFunctionBuilder
import org.testng.annotations.Test
import is.hail.asm4s._
import is.hail.utils.FastIndexedSeq
import org.apache.spark.sql.Row
class PSubsetStructSuite extends HailSuite {
  val debug = true

  @Test def testSubsetStruct() {
    val rt = PCanonicalStruct("a" -> PCanonicalString(), "b" -> PInt32(), "c" -> PInt64())
    val intInput = 3
    val longInput = 4L
    val fb = EmitFunctionBuilder[Region, Int, Long, Long]("fb")
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString("hello"),
        srvb.advance(),
        srvb.addInt(fb.getCodeParam[Int](2)),
        srvb.advance(),
        srvb.addLong(fb.getCodeParam[Long](3)),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, intInput, longInput))

    if (debug) {
      println(rv.pretty(rt))
    }

    val view = PSubsetStruct(rt, "a", "c")
    assert(view.size == 2)
    assert(view.field("a") == rt.field("a"))
    assert(view.field("c") == rt.field("c"))

    val rtV = SafeRow.read(rt, rv.offset).asInstanceOf[Row]
    val viewV = SafeRow.read(view, rv.offset).asInstanceOf[Row]

    assert(rtV(0)  == viewV(0) && rtV(2) == viewV(1))
  }

  @Test def testConstruction(): Unit = {
    val ps1 = PCanonicalStruct("a" -> PCanonicalArray(PInt32(true)), "b" -> PInt64())
    val ps2 = PCanonicalStruct("a" -> PCanonicalArray(PInt32(true)), "b" -> PInt64())

    val srcType = PSubsetStruct(ps1, "b")
    val destType = PSubsetStruct(ps2, "b")
    val srcValue = Annotation(IndexedSeq(1,5,7,2,31415926), 31415926535897L)
    val dstValue = Annotation(31415926535897L)
    PhysicalTestUtils.copyTestExecutor(srcType, destType, srcValue, deepCopy = false, interpret = true, expectedValue = dstValue)
  }
}
