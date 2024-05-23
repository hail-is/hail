package is.hail.expr.ir.agg

import is.hail.HailSuite
import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitFunctionBuilder}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.{PCanonicalArray, PCanonicalString}
import is.hail.types.physical.stypes.primitives.SFloat64Value

import org.testng.annotations.Test

class DownsampleSuite extends HailSuite {

  @Test def testLargeRandom(): Unit = {
    val lt = PCanonicalArray(PCanonicalString())
    val fb = EmitFunctionBuilder[RegionPool, Unit](ctx, "foo")
    val cb = fb.ecb
    val ds1 = new DownsampleState(cb, VirtualTypeWithReq(lt), maxBufferSize = 4)
    val ds2 = new DownsampleState(cb, VirtualTypeWithReq(lt), maxBufferSize = 4)
    val ds3 = new DownsampleState(cb, VirtualTypeWithReq(lt), maxBufferSize = 4)

    val stagedPool = fb.newLocal[RegionPool]("pool")

    val rng = fb.newRNG(0)
    val i = fb.newLocal[Int]()

    val x = fb.newLocal[Double]()
    val y = fb.newLocal[Double]()
    fb.emitWithBuilder { cb =>
      cb.assign(stagedPool, fb.getCodeParam[RegionPool](1))
      cb.assign(ds1.r, Region.stagedCreate(Region.SMALL, stagedPool))
      cb.assign(ds2.r, Region.stagedCreate(Region.SMALL, stagedPool))
      cb.assign(ds3.r, Region.stagedCreate(Region.SMALL, stagedPool))
      cb.assign(i, 0)
      ds1.init(cb, 100)
      ds2.init(cb, 100)
      cb.while_(
        i < 10000000, {
          cb.assign(x, rng.invoke[Double, Double, Double]("runif", 0d, 1d))
          cb.assign(y, rng.invoke[Double, Double, Double]("runif", 0d, 1d))

          ds1.insert(
            cb,
            EmitCode.present(cb.emb, new SFloat64Value(x)),
            EmitCode.present(cb.emb, new SFloat64Value(y)),
            EmitCode.missing(cb.emb, PCanonicalArray(PCanonicalString()).sType),
          )
          cb.assign(i, i + const(1))
        },
      )
      ds1.merge(cb, ds2)
      ds3.init(cb, 100)
      ds1.merge(cb, ds3)
      Code._empty
    }

    pool.scopedSmallRegion { r =>
      fb.resultWithIndex().apply(theHailClassLoader, ctx.fs, ctx.taskContext, r).apply(pool)
    }
  }
}
