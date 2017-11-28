package is.hail.expr.ir

import is.hail.annotations._
import is.hail.expr.{TAggregable, TArray}

object RunAggregators {
  // FIXME: don't use annotation interface
  def onArray(ir: IR, tAgg: TAggregable, region: MemoryBuffer, aOff: Long, elementToAnnotation: (MemoryBuffer, Long) => Annotation, extraBindings: (Int) => Array[Annotation]): (IR, Long) = {
    val tArray = TArray(tAgg.elementType)
    Infer(ir)
    val (post, agg) = ExtractAggregators(ir, tAgg)
    val perElementStart = region.offset
    var i = 0
    while (i < tArray.loadLength(region, aOff)) {
      val v = if (tArray.isElementDefined(region, aOff, i))
        elementToAnnotation(region, tArray.loadElement(region, aOff, i))
      else
        null
      agg.seqOp(region, tAgg.createCarrier(region, v, extraBindings(i):_*), false)
      // all allocated memory is unreachable because the aggregator value is
      // stored off-region
      region.offset = perElementStart
      i += 1
    }
    val outOff = agg.result(region)
    (post, outOff)
  }
}
