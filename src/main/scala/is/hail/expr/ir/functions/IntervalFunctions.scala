package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
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

    registerCodeWithMissingness("contains", TInterval(tv("T")), tv("T"), TBoolean()) {
      case (mb, interval, point) =>
        val pointType = tv("T").t
        val intervalType: TInterval = TInterval(pointType)

        val compare = mb.getCodeOrdering(pointType, CodeOrdering.compare, missingGreatest = true)
        val region = getRegion(mb)

        val pointMissing = mb.newLocal[Boolean]
        val pValue = mb.newLocal()(typeToTypeInfo(pointType))
        val iValue = mb.newLocal[Long]

        val startDef = intervalType.startDefined(region, iValue)
        val endDef = intervalType.endDefined(region, iValue)
        val start = region.getIRIntermediate(pointType)(intervalType.startOffset(iValue))
        val end = region.getIRIntermediate(pointType)(intervalType.endOffset(iValue))
        val includeStart = intervalType.includeStart(region, iValue)
        val includeEnd = intervalType.includeEnd(region, iValue)

        val cmp = mb.newLocal[Int]

        val contains = Code(
          iValue := interval.value[Long],
          pointMissing := point.m,
          pValue.storeAny(point.v),
          cmp := compare(region, (!startDef, start), region, (pointMissing, pValue)),
          (cmp > 0 || (cmp.ceq(0) && includeStart)) && Code(
            cmp := compare(region, (pointMissing, pValue), region, (!endDef, end)),
            cmp < 0 || (cmp.ceq(0) && includeEnd)))

        EmitTriplet(
          Code(interval.setup, point.setup),
          interval.m,
          contains)
    }

    registerCode("isEmpty", TInterval(tv("T")), TBoolean()) {
      case (mb, interval) =>
        val pointType = tv("T").t
        val intervalType: TInterval = TInterval(pointType)
        val gt = mb.getCodeOrdering(pointType, CodeOrdering.gt, missingGreatest = true)
        val gteq = mb.getCodeOrdering(pointType, CodeOrdering.gteq, missingGreatest = true)

        val iValue = mb.newLocal[Long]
        val region = getRegion(mb)

        val startDef = intervalType.startDefined(region, iValue)
        val endDef = intervalType.endDefined(region, iValue)
        val start = region.getIRIntermediate(pointType)(intervalType.startOffset(iValue))
        val end = region.getIRIntermediate(pointType)(intervalType.endOffset(iValue))
        val includeStart = intervalType.includeStart(region, iValue)
        val includeEnd = intervalType.includeEnd(region, iValue)
        Code(
          iValue := interval,
          (includeStart && includeEnd).mux(
            gt(region, (!startDef, start), region, (!endDef, end)),
            gteq(region, (!startDef, start), region, (!endDef, end)))
        )
    }

    registerCode("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean()) {
      case (mb, interval1, interval2) =>
        val pointType = tv("T").t
        val intervalType: TInterval = TInterval(pointType)

        val compare = mb.getCodeOrdering(pointType, CodeOrdering.compare, missingGreatest = true)
        val region = getRegion(mb)

        val iValue1 = mb.newLocal[Long]
        val iValue2 = mb.newLocal[Long]

        val startDef = intervalType.startDefined(region, iValue)
        val endDef = intervalType.endDefined(region, iValue)
        val start = region.getIRIntermediate(pointType)(intervalType.startOffset(iValue))
        val end = region.getIRIntermediate(pointType)(intervalType.endOffset(iValue))
        val includeStart = intervalType.includeStart(region, iValue)
        val includeEnd = intervalType.includeEnd(region, iValue)

        val cmp = mb.newLocal[Int]

        !Code(
          iValue1 := interval1,
          iValue2 := interval1,
          cmp := compare(region, (!startDef, start), region, (pointMissing, pValue)),
          (cmp > 0 || (cmp.ceq(0) && includeStart)) && Code(
            cmp := compare(region, (pointMissing, pValue), region, (!endDef, end)),
            cmp < 0 || (cmp.ceq(0) && includeEnd)))
    }
  }
}
