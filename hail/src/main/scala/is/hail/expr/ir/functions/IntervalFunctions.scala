package is.hail.expr.ir.functions

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir._
import is.hail.expr.types.physical.{PBoolean, PCanonicalInterval, PInterval, PType}
import is.hail.expr.types.virtual.{TArray, TBoolean, TInt32, TInterval}
import is.hail.utils._

object IntervalFunctions extends RegistryFunctions {

  def registerAll(): Unit = {

    registerCodeWithMissingness("Interval", tv("T"), tv("T"), TBoolean, TBoolean, TInterval(tv("T")),
      { case (startpt, endpt, includesStartPT, includesEndPT) =>
        PCanonicalInterval(
          InferPType.getNestedElementPTypes(Seq(startpt, endpt)),
          required = includesStartPT.required && includesEndPT.required
        )
      }) {
      case (r, rt, (startT, start), (endT, end), (includesStartT, includesStart), (includesEndT, includesEnd)) =>
        val srvb = new StagedRegionValueBuilder(r, rt)

        val mv = r.mb.newLocal[Boolean]()
        val vv = r.mb.newLocal[Long]()

        val ctor = Code(
          mv := includesStart.m || includesEnd.m,
          vv := 0L,
          mv.mux(
            Code._empty,
            Code(FastIndexedSeq(
              srvb.start(),
              start.m.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(startT)(start.v)),
              srvb.advance(),
              end.m.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(endT)(end.v)),
              srvb.advance(),
              srvb.addBoolean(includesStart.value[Boolean]),
              srvb.advance(),
              srvb.addBoolean(includesEnd.value[Boolean]),
              srvb.advance(),
              vv := srvb.offset))),
          Code._empty)

        EmitCode(
          Code(start.setup, end.setup, includesStart.setup, includesEnd.setup, ctor),
          mv,
          PCode(rt, vv))
    }

    registerCodeWithMissingness("start", TInterval(tv("T")), tv("T"),
      (x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (r, rt, (intervalT: PInterval, interval)) =>
        val iv = r.mb.newLocal[Long]()
        EmitCode(
          Code(interval.setup, iv.storeAny(defaultValue(intervalT))),
          interval.m || !Code(iv := interval.value[Long], intervalT.startDefined(iv)),
          PCode(rt, Region.loadIRIntermediate(intervalT.pointType)(intervalT.startOffset(iv)))
        )
    }

    registerCodeWithMissingness("end", TInterval(tv("T")), tv("T"),
      (x: PType) => x.asInstanceOf[PInterval].pointType.orMissing(x.required)) {
      case (r, rt, (intervalT: PInterval, interval)) =>
        val iv = r.mb.newLocal[Long]()
        EmitCode(
          Code(interval.setup, iv.storeAny(defaultValue(intervalT))),
          interval.m || !Code(iv := interval.value[Long], intervalT.endDefined(iv)),
          PCode(rt, Region.loadIRIntermediate(intervalT.pointType)(intervalT.endOffset(iv)))
        )
    }

    registerCode("includesStart", TInterval(tv("T")), TBoolean, (x: PType) =>
      PBoolean(x.required)
    ) {
      case (r, rt, (intervalT: PInterval, interval: Code[Long])) =>
        intervalT.includesStart(interval)
    }

    registerCode("includesEnd", TInterval(tv("T")), TBoolean, (x: PType) =>
      PBoolean(x.required)
    ) {
      case (r, rt, (intervalT: PInterval, interval: Code[Long])) =>
        intervalT.includesEnd(interval)
    }

    registerCodeWithMissingness("contains", TInterval(tv("T")), tv("T"), TBoolean, {
      case(intervalT: PInterval, _: PType) => PBoolean(intervalT.required)
    }) {
      case (r, rt, (intervalT: PInterval, intTriplet), (pointT, pointTriplet)) =>
        val mPoint = r.mb.newLocal[Boolean]()
        val vPoint = r.mb.newLocal()(typeToTypeInfo(pointT))

        val cmp = r.mb.newLocal[Int]()
        val interval = new IRInterval(r, intervalT, intTriplet.value[Long])
        val compare = interval.ordering(CodeOrdering.compare)

        val contains = Code(
          interval.storeToLocal,
          mPoint := pointTriplet.m,
          vPoint.storeAny(pointTriplet.v),
          cmp := compare((mPoint, vPoint), interval.start),
          (cmp > 0 || (cmp.ceq(0) && interval.includesStart)) && Code(
            cmp := compare((mPoint, vPoint), interval.end),
            cmp < 0 || (cmp.ceq(0) && interval.includesEnd)))

        EmitCode(
          Code(intTriplet.setup, pointTriplet.setup),
          intTriplet.m,
          PCode(rt, contains))
    }

    registerCode("isEmpty", TInterval(tv("T")), TBoolean, (pt: PType) => PBoolean(pt.required)) {
      case (r, rt, (intervalT: PInterval, intOff)) =>
        val interval = new IRInterval(r, intervalT, intOff)

        Code(
          interval.storeToLocal,
          interval.isEmpty
        )
    }

    registerCode("overlaps", TInterval(tv("T")), TInterval(tv("T")), TBoolean, {
      case(i1t: PType, i2t: PType) =>
        PBoolean(i1t.required && i2t.required)
    }) {
      case (r, rt, (i1t: PInterval, iOff1), (i2t: PInterval, iOff2)) =>
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

    registerIR("sortedNonOverlappingIntervalsContain",
      TArray(TInterval(tv("T"))), tv("T"), TBoolean) { case (intervals, value) =>
      val uid = genUID()
      val uid2 = genUID()
      Let(uid, LowerBoundOnOrderedCollection(intervals, value, onKey = true),
        (Let(uid2, Ref(uid, TInt32) - I32(1), (Ref(uid2, TInt32) >= 0)
          && invoke("contains", TBoolean, ArrayRef(intervals, Ref(uid2, TInt32)), value)))
          || ((Ref(uid, TInt32) < ArrayLen(intervals))
          && invoke("contains", TBoolean, ArrayRef(intervals, Ref(uid, TInt32)), value)))
    }
  }
}

class IRInterval(r: EmitRegion, typ: PInterval, value: Code[Long]) {
  val ref: LocalRef[Long] = r.mb.newLocal[Long]()
  val region: Code[Region] = r.region

  def ordering(op: CodeOrdering.Op): ((Code[Boolean], Code[_]), (Code[Boolean], Code[_])) => Code[op.ReturnType] =
    r.mb.getCodeOrdering(typ.pointType, op)(_, _)

  def storeToLocal: Code[Unit] = ref := value

  def start: (Code[Boolean], Code[_]) =
    (!typ.startDefined(ref), Region.getIRIntermediate(typ.pointType)(typ.startOffset(ref)))
  def end: (Code[Boolean], Code[_]) =
    (!typ.endDefined(ref), Region.getIRIntermediate(typ.pointType)(typ.endOffset(ref)))
  def includesStart: Code[Boolean] = typ.includesStart(ref)
  def includesEnd: Code[Boolean] = typ.includesEnd(ref)

  def isEmpty: Code[Boolean] = {
    val gt = ordering(CodeOrdering.gt)
    val gteq = ordering(CodeOrdering.gteq)

    (includesStart && includesEnd).mux(
      gt(start, end),
      gteq(start, end))
  }

  def isAboveOnNonempty(other: IRInterval): Code[Boolean] = {
    val cmp = r.mb.newLocal[Int]()
    val compare = ordering(CodeOrdering.compare)
    Code(
      cmp := compare(start, other.end),
      cmp > 0 || (cmp.ceq(0) && (!includesStart || !other.includesEnd)))
  }

  def isBelowOnNonempty(other: IRInterval): Code[Boolean] = {
    val cmp = r.mb.newLocal[Int]()
    val compare = ordering(CodeOrdering.compare)
    Code(
      cmp := compare(end, other.start),
      cmp < 0 || (cmp.ceq(0) && (!includesEnd || !other.includesStart)))
  }
}
