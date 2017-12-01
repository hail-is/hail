package is.hail.expr.ir

import is.hail.annotations._
import is.hail.expr.{TAggregable, TArray, RegionValueAggregator}

object RunAggregators {
  def onArray(ir: IR, tAgg: TAggregable, aOff: Long,
    v0: Long, m0: Boolean): (IR, (MemoryBuffer) => Long) =
    genericOnArray(ir, tAgg, aOff, { (agg, region, e, me, i) =>
      agg.seqOp(region, tAgg.createScopeCarrier(region,
        e, me,
        v0, m0), false)
    })

  def onArray(ir: IR, tAgg: TAggregable, aOff: Long,
    v0: Long, m0: Boolean,
    v1: Long, m1: Boolean): (IR, (MemoryBuffer) => Long) =
    genericOnArray(ir, tAgg, aOff, { (agg, region, e, me, i) =>
      agg.seqOp(region, tAgg.createScopeCarrier(region,
        e, me,
        v0, m0, v1, m1), false)
    })

  def onArray(ir: IR, tAgg: TAggregable, aOff: Long,
    v0: Long, m0: Boolean,
    v1: Long, m1: Boolean,
    v2: Long, m2: Boolean): (IR, (MemoryBuffer) => Long) =
    genericOnArray(ir, tAgg, aOff, { (agg, region, e, me, i) =>
      agg.seqOp(region, tAgg.createScopeCarrier(region,
        e, me,
        v0, m0, v1, m1, v2, m2), false)
    })

  def onArray(ir: IR, tAgg: TAggregable, aOff: Long,
    v0: Long, m0: Boolean,
    v1: Long, m1: Boolean,
    v2: Long, m2: Boolean,
    v3: Long, m3: Boolean): (IR, (MemoryBuffer) => Long) =
    genericOnArray(ir, tAgg, aOff, { (agg, region, e, me, i) =>
      agg.seqOp(region, tAgg.createScopeCarrier(region,
        e, me,
        v0, m0, v1, m1, v2, m2, v3, m3), false)
    })

  def onArray(ir: IR, tAgg: TAggregable, aOff: Long,
    v0: Long, m0: Boolean,
    v1: Long, m1: Boolean,
    v2: Long, m2: Boolean,
    v3: Long, m3: Boolean,
    v4: Long, m4: Boolean): (IR, (MemoryBuffer) => Long) =
    genericOnArray(ir, tAgg, aOff, { (agg, region, e, me, i) =>
      agg.seqOp(region, tAgg.createScopeCarrier(region,
        e, me,
        v0, m0, v1, m1, v2, m2, v3, m3, v4, m4), false)
    })

  private def genericOnArray(ir: IR, tAgg: TAggregable, aOff: Long, seqOp: (RegionValueAggregator, MemoryBuffer, Long, Boolean, Int) => Unit): (IR, (MemoryBuffer) => Long) = {
    val tArray = TArray(tAgg.elementType)
    Infer(ir, Some(tAgg))
    val (post, agg) = ExtractAggregators(ir, tAgg)
    val runAggregation = { (region: MemoryBuffer) =>
      val perElementStart = region.size
      var i = 0
      while (i < tArray.loadLength(region, aOff)) {
        val me = tArray.isElementMissing(region, aOff, i)
        val e = if (me) 0 else tArray.loadElement(region, aOff, i)
        seqOp(agg, region, e, me, i)
        // all allocated memory is unreachable because the aggregator value is
        // stored off-region
        region.clear(perElementStart)
        i += 1
      }
      agg.result(region)
    }
    (post, runAggregation)
  }
}
