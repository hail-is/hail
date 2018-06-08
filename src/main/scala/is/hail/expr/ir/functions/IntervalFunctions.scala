package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir._
import is.hail.expr.{Apply, ir, types}
import is.hail.expr.types.{TBoolean, TBooleanOptional, TInterval, TStruct}
import is.hail.utils._

object IntervalFunctions extends RegistryFunctions {

  def registerAll(): Unit = {

    registerCodeWithMissingness("Interval", tv("T"), tv("T"), TBoolean(), TBoolean(), TInterval(tv("T"))) {
      (mb, start, end, includeStart, includeEnd) =>
        val srvb = new StagedRegionValueBuilder(mb, TInterval(tv("T").t))
        val missing = includeStart.m || includeEnd.m
        val value = Code(
          srvb.start(),
          start.m.mux(
            srvb.setMissing(),
            srvb.addIRIntermediate(tv("T").t)(start.v)),
          srvb.advance(),
          end.m.mux(
            srvb.setMissing(),
            srvb.addIRIntermediate(tv("T").t)(end.v)),
          srvb.advance(),
          srvb.addBoolean(includeStart.value[Boolean]),
          srvb.advance(),
          srvb.addBoolean(includeEnd.value[Boolean]),
          srvb.advance(),
          srvb.offset
        )

        EmitTriplet(
          Code(start.setup, end.setup, includeStart.setup, includeEnd.setup),
          missing,
          value)
    }

    registerCodeWithMissingness("start", TInterval(tv("T")), tv("T")) {
      case (mb, interval) =>
        val tinterval = TInterval(tv("T").t)
        val region = mb.getArg[Region](1).load()
        val iv = mb.newLocal[Long]
        EmitTriplet(
          interval.setup,
          interval.m || !Code(iv := interval.value[Long], tinterval.startDefined(region, iv)),
          region.loadIRIntermediate(tv("T").t)(tinterval.startOffset(iv))
        )
    }

    registerCodeWithMissingness("end", TInterval(tv("T")), tv("T")) {
      case (mb, interval) =>
        val tinterval = TInterval(tv("T").t)
        val region = mb.getArg[Region](1).load()
        val iv = mb.newLocal[Long]
        EmitTriplet(
          interval.setup,
          interval.m || !Code(iv := interval.value[Long], tinterval.endDefined(region, iv)),
          region.loadIRIntermediate(tv("T").t)(tinterval.endOffset(iv))
        )
    }

    registerCode("includesStart", TInterval(tv("T")), TBooleanOptional) {
      case (mb, interval: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        TInterval(tv("T").t).includeStart(region, interval)
    }

    registerCode("includesEnd", TInterval(tv("T")), TBooleanOptional) {
      case (mb, interval: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        TInterval(tv("T").t).includeEnd(region, interval)
    }
  }
}
