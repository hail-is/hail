package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.check.{Gen, Prop}
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion}
import is.hail.expr.types._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PContainerTest extends HailSuite {


    val showRVInfo = true

    @Test
    def testString() {
      val rt = PString()
      val input = "hello"
      val fb = FunctionBuilder.functionBuilder[Region, String, Long]
      val srvb = new StagedRegionValueBuilder(fb, rt)

      fb.emit(
        Code(
          srvb.start(),
          srvb.addString(fb.getArg[String](2)),
          srvb.end()
        )
      )

      val region = Region()
      val rv = RegionValue(region)
      rv.setOffset(fb.result()()(region, input))

      if (showRVInfo) {
        printRegion(region, "string")
        println(rv.pretty(rt))
      }

      val region2 = Region()
      val rv2 = RegionValue(region2)
      val bytes = input.getBytes()
      val boff = PBinary.allocate(region2, bytes.length)
      Region.storeInt(boff, bytes.length)
      Region.storeBytes(PBinary.bytesOffset(boff), bytes)
      rv2.setOffset(boff)

      if (showRVInfo) {
        printRegion(region2, "string")
        println(rv2.pretty(rt))
      }

      assert(rv.pretty(rt) == rv2.pretty(rt))
      assert(PString.loadString(rv.region, rv.offset) ==
        PString.loadString(rv2.region, rv2.offset))
    }
}
