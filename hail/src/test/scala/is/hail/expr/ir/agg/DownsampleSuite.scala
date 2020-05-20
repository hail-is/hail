package is.hail.expr.ir.agg

import is.hail.HailSuite
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.types.physical.{PCanonicalArray, PCanonicalString}
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.Test

class DownsampleSuite extends HailSuite {

  @Test def testLargeRandom(): Unit = {
    val lt = PCanonicalArray(PCanonicalString())
    val fb = EmitFunctionBuilder[Unit](ctx, "foo")
    val cb = fb.ecb
    val ds1 = new DownsampleState(cb, lt, maxBufferSize = 4)
    val ds2 = new DownsampleState(cb, lt, maxBufferSize = 4)
    val ds3 = new DownsampleState(cb, lt, maxBufferSize = 4)

    val rng = fb.newRNG(0)
    val i = fb.newLocal[Int]()

    val x = fb.newLocal[Double]()
    val y = fb.newLocal[Double]()
    fb.emitWithBuilder { cb =>
      cb.assign(ds1.r, Region.stagedCreate(Region.SMALL))
      cb.assign(ds2.r, Region.stagedCreate(Region.SMALL))
      cb.assign(ds3.r, Region.stagedCreate(Region.SMALL))
      cb.assign(i, 0)
      cb += ds1.init(100)
      cb += ds2.init(100)
      cb.whileLoop(i < 10000000, {
          cb.assign(x, rng.invoke[Double, Double, Double]("runif", 0d, 1d))
          cb.assign(y, rng.invoke[Double, Double, Double]("runif", 0d, 1d))
          cb += ds1.insert(x, y, true, 0L)
          cb.assign(i, i + const(1))
      })
      ds1.merge(cb, ds2)
      cb += ds3.init(100)
      ds1.merge(cb, ds3)
      Code._empty
    }

    Region.smallScoped { r =>
      fb.resultWithIndex().apply(0, r).apply()
    }
  }
}
