package is.hail.expr.ir.agg

import is.hail.HailSuite
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.expr.types.physical.{PArray, PString}
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.Test

class DownsampleSuite extends HailSuite {

  @Test def testLargeRandom(): Unit = {
    val lt = PArray(PString())
    val fb = EmitFunctionBuilder[Unit]("foo")
    val cb = fb.ecb
    val ds1 = new DownsampleState(cb, lt, maxBufferSize = 4)
    val ds2 = new DownsampleState(cb, lt, maxBufferSize = 4)
    val ds3 = new DownsampleState(cb, lt, maxBufferSize = 4)

    val rng = fb.newRNG(0)
    val i = fb.newLocal[Int]()

    val x = fb.newLocal[Double]()
    val y = fb.newLocal[Double]()
    fb.emit(Code(FastIndexedSeq(
      ds1.r := Region.stagedCreate(Region.SMALL),
      ds2.r := Region.stagedCreate(Region.SMALL),
      ds3.r := Region.stagedCreate(Region.SMALL),
      i := 0,
      ds1.init(100),
      ds2.init(100),
      Code.whileLoop(i < 10000000,
        x := rng.invoke[Double, Double, Double]("runif", 0d, 1d),
        y := rng.invoke[Double, Double, Double]("runif", 0d, 1d),
        ds1.insert(x, y, true, 0L),
        i := i + const(1)),
      ds1.merge(ds2),
      ds3.init(100),
      ds1.merge(ds3)
    )))

    Region.smallScoped { r =>
      fb.resultWithIndex().apply(0, r).apply()
    }
  }
}
