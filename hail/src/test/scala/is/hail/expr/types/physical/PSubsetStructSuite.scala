package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Region, RegionValue, StagedRegionValueBuilder}
import is.hail.asm4s.Code
import is.hail.expr.ir.EmitFunctionBuilder
import org.testng.annotations.Test
import is.hail.asm4s._
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
    println(s"view: ${view.size}")
    assert(view.size == 2)
    assert(Region.loadInt(rt.loadField(rv.offset, 0)) == Region.loadInt(view.loadField(rv.offset, 0)))
    assert(Region.loadInt(rt.loadField(rv.offset, 2)) == Region.loadInt(view.loadField(rv.offset, 1)))

    assert(Region.loadInt(rt.loadField(rv.offset, "a")) == Region.loadInt(view.loadField(rv.offset, "a")))
    assert(Region.loadInt(rt.loadField(rv.offset, "c")) == Region.loadInt(view.loadField(rv.offset, "c")))

    val view2 = view.selectFields(Seq("c"))
    assert(view2.size == 1)
    assert(Region.loadInt(rt.loadField(rv.offset, 2)) == Region.loadInt(view2.loadField(rv.offset, 0)))

    val view3 = view.dropFields(Set("c"))
    assert(view3.size == 1)
    assert(Region.loadInt(rt.loadField(rv.offset, 0)) == Region.loadInt(view3.loadField(rv.offset, 0)))
  }
}
