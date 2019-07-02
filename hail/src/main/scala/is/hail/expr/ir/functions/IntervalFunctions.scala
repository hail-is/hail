package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir._
import is.hail.expr.types.physical.{PInterval, PType}
import is.hail.expr.types.virtual.{TBoolean, TBooleanOptional, TInterval}
import is.hail.utils._

object IntervalFunctions extends RegistryFunctions {

  def registerAll(): Unit = {

    registerCodeWithMissingness("Interval", tv("T"), tv("T"), TBoolean(), TBoolean(), TInterval(tv("T"))) {
      case (r, (startT, start), (endT, end), (includeStartT, includeStart), (includeEndT, includeEnd)) =>
        val srvb = new StagedRegionValueBuilder(r, PInterval(PType.canonical(startT)))

        val mv = r.mb.newLocal[Boolean]
        val vv = r.mb.newLocal[Long]

        val ctor = Code(
          mv := includeStart.m || includeEnd.m,
          vv := 0L,
          mv.mux(
            Code._empty,
            Code(
              srvb.start(),
              start.m.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(startT)(start.v)),
              srvb.advance(),
              end.m.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(endT)(end.v)),
              srvb.advance(),
              srvb.addBoolean(includeStart.value[Boolean]),
              srvb.advance(),
              srvb.addBoolean(includeEnd.value[Boolean]),
              srvb.advance(),
              vv := srvb.offset)),
          Code._empty[Unit])

        EmitTriplet(
          Code(start.setup, end.setup, includeStart.setup, includeEnd.setup, ctor),
          mv,
          vv)
    }

    registerCodeWithMissingness("start", TInterval(tv("T")), tv("T")) {
      case (r, (intervalT: PInterval, interval)) =>
        val region = r.region
        val iv = r.mb.newLocal[Long]
        EmitTriplet(
          Code(interval.setup, iv.storeAny(defaultValue(intervalT))),
          interval.m || !Code(iv := interval.value[Long], intervalT.startDefined(region, iv)),
          region.loadIRIntermediate(tv("T").t)(intervalT.startOffset(iv))
        )
    }

    registerCodeWithMissingness("end", TInterval(tv("T")), tv("T")) {
      case (r, (intervalT: PInterval, interval)) =>
        val region = r.region
        val iv = r.mb.newLocal[Long]
        EmitTriplet(
          Code(interval.setup, iv.storeAny(defaultValue(intervalT))),
          interval.m || !Code(iv := interval.value[Long], intervalT.endDefined(region, iv)),
          region.loadIRIntermediate(tv("T").t)(intervalT.endOffset(iv))
        )
    }

    registerCode("includesStart", TInterval(tv("T")), TBooleanOptional) {
      case (r, (intervalT: PInterval, interval: Code[Long])) =>
        intervalT.includeStart(r.region, interval)
    }

    registerCode("includesEnd", TInterval(tv("T")), TBooleanOptional) {
      case (r, (intervalT: PInterval, interval: Code[Long])) =>
        intervalT.includeEnd(r.region, interval)
    }

    registerCodeWithMissingness("contains", TInterval(tv("T")), tv("T"), TBoolean()) {
      case (r, (intervalT: PInterval, intTriplet), (pointT, pointTriplet)) =>
        val pointType = tv("T").t.physicalType

        val mPoint = r.mb.newLocal[Boolean]
        val vPoint = r.mb.newLocal()(typeToTypeInfo(pointType))

        val cmp = r.mb.newLocal[Int]
        val interval = new IRInterval(r, intervalT, intTriplet.value[Long])
        val compare = interval.ordering(CodeOrdering.compare)

        val contains = Code(
          interval.storeToLocal,
          mPoint := pointTriplet.m,
          vPoint.storeAny(pointTriplet.v),
          cmp := compare((mPoint, vPoint), interval.start),
          (cmp > 0 || (cmp.ceq(0) && interval.includeStart)) && Code(
            cmp := compare((mPoint, vPoint), interval.end),
            cmp < 0 || (cmp.ceq(0) && interval.includeEnd)))

        EmitTriplet(
          Code(intTriplet.setup, pointTriplet.setup),
          intTriplet.m,
          contains)
    }

    registerCode("isEmpty", TInterval(tv("T")), TBoolean()) {
      case (r, (intervalT: PInterval, intOff)) =>
        val interval = new IRInterval(r, intervalT, intOff)

        Code(
          interval.storeToLocal,
          interval.isEmpty
        )
    }

    registerCode("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean()) {
      case (r, (i1t: PInterval, iOff1), (i2t: PInterval, iOff2)) =>
        val interval1 = new IRInterval(r, i1t, iOff1)
        val interval2 = new IRInterval(r, i2t, iOff2)

        Code(
          interval1.storeToLocal,
          interval2.storeToLocal,
          !(interval1.isEmpty || interval2.isEmpty ||
            interval1.isBelowOnNonempty(interval2) ||
            interval1.isAboveOnNonempty(interval2))
        )
    }
  }
}

class IRInterval(r: EmitRegion, typ: PInterval, value: Code[Long]) {
  val ref: LocalRef[Long] = r.mb.newLocal[Long]
  val region: Code[Region] = r.region

  def ordering[T](op: CodeOrdering.Op): ((Code[Boolean], Code[_]), (Code[Boolean], Code[_])) => Code[T] =
    r.mb.getCodeOrdering[T](typ.pointType, op)(region, _, region, _)

  def storeToLocal: Code[Unit] = ref := value

  def start: (Code[Boolean], Code[_]) =
    (!typ.startDefined(region, ref), region.getIRIntermediate(typ.pointType)(typ.startOffset(ref)))
  def end: (Code[Boolean], Code[_]) =
    (!typ.endDefined(region, ref), region.getIRIntermediate(typ.pointType)(typ.endOffset(ref)))
  def includeStart: Code[Boolean] = typ.includeStart(region, ref)
  def includeEnd: Code[Boolean] = typ.includeEnd(region, ref)

  def isEmpty: Code[Boolean] = {
    val gt = ordering(CodeOrdering.gt)
    val gteq = ordering(CodeOrdering.gteq)

    (includeStart && includeEnd).mux(
      gt(start, end),
      gteq(start, end))
  }

  def isAboveOnNonempty(other: IRInterval): Code[Boolean] = {
    val cmp = r.mb.newLocal[Int]
    val compare = ordering(CodeOrdering.compare)
    Code(
      cmp := compare(start, other.end),
      cmp > 0 || (cmp.ceq(0) && (!includeStart || !other.includeEnd)))
  }

  def isBelowOnNonempty(other: IRInterval): Code[Boolean] = {
    val cmp = r.mb.newLocal[Int]
    val compare = ordering(CodeOrdering.compare)
    Code(
      cmp := compare(end, other.start),
      cmp < 0 || (cmp.ceq(0) && (!includeEnd || !other.includeStart)))
  }
}
