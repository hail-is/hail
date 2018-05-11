package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.expr.types.coerce

object AggregatorFunctions extends RegistryFunctions {
  def registerAll() {
    registerIR("sum", TAggregable(TInt64()))(ApplyAggOp(_, Sum(), FastSeq()))
    registerIR("sum", TAggregable(TFloat64()))(ApplyAggOp(_, Sum(), FastSeq()))

    registerIR("product", TAggregable(TInt64()))(ApplyAggOp(_, Product(), FastSeq()))
    registerIR("product", TAggregable(TFloat64()))(ApplyAggOp(_, Product(), FastSeq()))

    registerIR("max", TAggregable(tnum("T")))(ApplyAggOp(_, Max(), FastSeq()))

    registerIR("min", TAggregable(tnum("T")))(ApplyAggOp(_, Min(), FastSeq()))

    registerIR("count", TAggregable(tv("T"))) { agg =>
      ApplyAggOp(AggMap(agg, "_", I32(0)), Count(), FastSeq())
    }

    registerIR("fraction", TAggregable(TBoolean())) { agg =>
      ApplyAggOp(agg, Fraction(), FastSeq())
    }

    registerIR("hist",
      TAggregable(TFloat64()), TFloat64(), TFloat64(), TInt32()
    ) { (agg, start, end, nbins) =>
      ApplyAggOp(agg, Histogram(), FastSeq(start, end, nbins))
    }
  }
}
